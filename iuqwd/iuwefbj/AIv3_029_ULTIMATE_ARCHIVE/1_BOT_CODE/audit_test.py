"""
AUDIT TEST — AIv3_029 Walk-Forward Code

4 tests:
  1. COIN-FLIP CONTROL: Replace AI with random coin flip. If still profitable,
     the edge isn't from the AI.
  2. NO-AI BASELINE: Remove AI entirely (take every candidate). If still
     profitable, the edge is from the risk management + momentum bars.
  3. OUT-OF-SAMPLE ASSETS: Test on assets NOT in the training set
     (LTCUSDT, LINKUSDT, DOTUSDT, etc.) to verify generalization.
  4. POSITION SIZING AUDIT: Track cur_risk and wallet across all trades
     to verify no runaway compounding.

BUGS FOUND IN CODE AUDIT:
  Bug A: Trailing stop is BROKEN. pos["best_price"] is never updated — it
  always stays at entry_price. The trailing logic uses a local variable that
  resets each bar. This means trailing stops DON'T WORK. (Conservative bug —
  makes results WORSE, not better.)

  Bug B: pos["sp"] is never updated by trailing. The stop is always the
  original stop (0.5 ATR below entry). No breakeven move, no trailing.
  (Same as Bug A — conservative, makes results worse.)

  Bug C: risk_amount = eff_risk * wallet. This uses pre-commission wallet.
  Slightly over-risks by the commission amount. (Minor.)

  Bug D: NO LOOKAHEAD FOUND. Cross-asset features use searchsorted to find
  the last bar BEFORE time t. Indicator values at past indices are identical
  regardless of future data in the dataframe. This is correct.
"""
import sys, os, time, json, gc, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from backtester import Costs, ema, rsi, atr, adx, bollinger
from smart_strategies import zscore, taker_buy_ratio
from bots_100 import rolling_linreg_slope, INITIAL_WALLET, COSTS, DATA_DIR, DOWNLOAD
from risk_managed_v2 import RiskConfig

# Import everything from the walkforward script
from walkforward_3year import (
    CONFIG, FEATURE_COLS, REF_SYMBOLS,
    load_data, get_indicators, get_cross_asset_features,
    extract_features, cand_momentum, generate_training_samples,
    make_risk_cfg, backtest_month, generate_test_months,
)
import walkforward_3year as wf
_IND_CACHE = wf._IND_CACHE

DOWNLOAD = Path("/home/z/my-project/download")

# Test window: last 6 months only (for speed)
TEST_MONTHS_6 = generate_test_months()[-6:]


def run_coinflip_control():
    """TEST 1: Replace AI with coin flip. Same everything else."""
    print("\n" + "="*100)
    print("TEST 1: COIN-FLIP CONTROL — Replace AI with random 50/50 decision")
    print("="*100)
    risk_cfg = make_risk_cfg()
    results = []

    for mi, month in enumerate(TEST_MONTHS_6):
        print(f"\n  Month {mi+1}/6: {month['label']}", flush=True)

        # Build coin-flip "AI" for each symbol
        # The AI gatekeeper in backtest_month calls ai.predict_proba(X)
        # and checks if proba >= threshold. For coin flip, we override
        # to return a random probability.
        class CoinFlipAI:
            def predict_proba(self, X):
                n = X.shape[0]
                proba = np.random.rand(n, 2)
                proba[:, 0] = 1 - proba[:, 1]
                return proba

        ai_per_symbol = {}
        threshold_per_symbol = {}
        for sym in CONFIG["symbols"]:
            ai_per_symbol[sym] = CoinFlipAI()
            threshold_per_symbol[sym] = 0.5  # 50% = coin flip

        trades, stats = backtest_month(
            ai_per_symbol, threshold_per_symbol, risk_cfg,
            month['test_start'], month['test_end']
        )
        print(f"    {stats['n_trades']} trades, WR={stats['win_rate']*100:.1f}%, "
              f"net=${stats['net']:+,.2f} ({stats['net_pct']*100:+.2f}%)", flush=True)
        results.append({"month": month['label'], **stats})
        del ai_per_symbol; gc.collect()

    n_prof = sum(1 for r in results if r['net'] > 0)
    avg_net = np.mean([r['net'] for r in results])
    avg_wr = np.mean([r['win_rate'] for r in results if r['n_trades'] > 0])
    print(f"\n  COIN FLIP RESULT: {n_prof}/6 profitable, avg=${avg_net:+,.2f}, avg WR={avg_wr*100:.1f}%")
    return results


def run_no_ai_baseline():
    """TEST 2: Remove AI entirely. Take EVERY candidate signal."""
    print("\n" + "="*100)
    print("TEST 2: NO-AI BASELINE — Take every candidate signal (no AI filter)")
    print("="*100)
    risk_cfg = make_risk_cfg()
    results = []

    for mi, month in enumerate(TEST_MONTHS_6):
        print(f"\n  Month {mi+1}/6: {month['label']}", flush=True)

        # No AI — pass empty dicts. The backtest code checks `if ai is not None`
        # and if not, sets proba=1.0 and always takes the trade.
        ai_per_symbol = {}
        threshold_per_symbol = {}

        trades, stats = backtest_month(
            ai_per_symbol, threshold_per_symbol, risk_cfg,
            month['test_start'], month['test_end']
        )
        print(f"    {stats['n_trades']} trades, WR={stats['win_rate']*100:.1f}%, "
              f"net=${stats['net']:+,.2f} ({stats['net_pct']*100:+.2f}%)", flush=True)
        results.append({"month": month['label'], **stats})
        gc.collect()

    n_prof = sum(1 for r in results if r['net'] > 0)
    avg_net = np.mean([r['net'] for r in results])
    avg_wr = np.mean([r['win_rate'] for r in results if r['n_trades'] > 0])
    print(f"\n  NO-AI RESULT: {n_prof}/6 profitable, avg=${avg_net:+,.2f}, avg WR={avg_wr*100:.1f}%")
    return results


def run_out_of_sample():
    """TEST 3: Test on assets NOT in the training set.
    Fetch LTC, LINK, DOT, MATIC, UNI, ATOM — 6 new assets.
    Train AI on the 9 original assets, test on new assets."""
    print("\n" + "="*100)
    print("TEST 3: OUT-OF-SAMPLE ASSETS — Test on 6 assets NOT in training")
    print("="*100)

    # Fetch new assets
    NEW_SYMBOLS = ["LTCUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT", "UNIUSDT", "ATOMUSDT"]
    import requests
    URL = "https://api.binance.com/api/v3/klines"
    KLINE_COLS = ["open_time","open","high","low","close","volume",
                  "close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","_"]

    for sym in NEW_SYMBOLS:
        cache = DATA_DIR / f"{sym}_1h.parquet"
        if cache.exists():
            print(f"  [cache] {sym}", flush=True)
            continue
        print(f"  Fetching {sym}...", flush=True)
        try:
            rows = []
            cur = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)
            end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
            session = requests.Session()
            while cur < end:
                r = session.get(URL, params={"symbol": sym, "interval": "1h",
                    "startTime": cur, "endTime": end, "limit": 1000}, timeout=30)
                if r.status_code in (429, 418):
                    import time as _t; _t.sleep(5); continue
                data = r.json()
                if not data: break
                rows.extend(data)
                cur = data[-1][6] + 1
                import time as _t; _t.sleep(0.12)
            df = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns=["_"])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
                df[c] = pd.to_numeric(df[c]).astype("float64")
            df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
            df.to_parquet(cache, index=False)
            print(f"    {sym}: {len(df)} rows", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)

    # Train AI on ORIGINAL 9 assets, test on NEW 6 assets
    risk_cfg = make_risk_cfg()
    test_month = TEST_MONTHS_6[-1]  # Just last month for speed
    print(f"\n  Testing on: {test_month['label']}")
    print(f"  Training on: 9 original assets")
    print(f"  Testing on: 6 NEW assets (LTC, LINK, DOT, MATIC, UNI, ATOM)")

    # Train AIs on original assets
    ai_per_symbol = {}
    threshold_per_symbol = {}
    for sym in CONFIG["symbols"]:
        try:
            samples = generate_training_samples(sym, test_month['train_start'], test_month['train_end'])
            if len(samples) < 30: continue
            X = np.array([[s[c] for c in FEATURE_COLS] for s in samples])
            y = np.array([s["label"] for s in samples])
            model = RandomForestClassifier(
                n_estimators=80, max_depth=5, min_samples_leaf=15,
                random_state=42, class_weight="balanced")
            model.fit(X, y)
            ai_per_symbol[sym] = model
            # Validate
            val_samples = generate_training_samples(sym, test_month['val_start'], test_month['val_end'])
            if len(val_samples) >= 10:
                X_val = np.array([[s[c] for c in FEATURE_COLS] for s in val_samples])
                y_val = np.array([s["label"] for s in val_samples])
                probas = model.predict_proba(X_val)[:, 1]
                best_t = 0.5; best_e = -999
                for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
                    mask = probas >= t
                    if mask.sum() < 3: continue
                    wr = y_val[mask].mean()
                    e = 3 * wr - 1
                    if e > best_e: best_e = e; best_t = t
                threshold_per_symbol[sym] = best_t
            else:
                threshold_per_symbol[sym] = 0.5
            print(f"    {sym}: trained on {len(samples)} samples, threshold={threshold_per_symbol[sym]:.2f}", flush=True)
        except Exception as e:
            print(f"    {sym}: FAILED - {e}", flush=True)

    # Test on NEW assets — use one of the trained AIs (ETH as proxy)
    # since we don't have AI trained on the new assets
    oos_config = {
        "symbols": NEW_SYMBOLS,
        "candidate_gen": "momentum",
        "max_concurrent": 1,
        "max_same_direction": 3,
    }
    # Use ETH's AI as a proxy for all new assets
    ai_oos = {}
    threshold_oos = {}
    for sym in NEW_SYMBOLS:
        ai_oos[sym] = ai_per_symbol.get("ETHUSDT")
        threshold_oos[sym] = threshold_per_symbol.get("ETHUSDT", 0.5)

    # Temporarily override CONFIG
    original_symbols = CONFIG["symbols"]
    CONFIG["symbols"] = NEW_SYMBOLS

    trades, stats = backtest_month(
        ai_oos, threshold_oos, risk_cfg,
        test_month['test_start'], test_month['test_end']
    )
    CONFIG["symbols"] = original_symbols  # restore

    print(f"\n  OOS RESULT: {stats['n_trades']} trades, WR={stats['win_rate']*100:.1f}%, "
          f"net=${stats['net']:+,.2f} ({stats['net_pct']*100:+.2f}%)")
    return trades, stats


def run_position_sizing_audit():
    """TEST 4: Track cur_risk and wallet across all trades in one month.
    Verify cur_risk stays within [0.4%, 1.5%] and doesn't runaway."""
    print("\n" + "="*100)
    print("TEST 4: POSITION SIZING AUDIT — Track cur_risk and wallet")
    print("="*100)

    risk_cfg = make_risk_cfg()
    test_month = TEST_MONTHS_6[-1]
    print(f"  Month: {test_month['label']}")

    # Train AI for one symbol (ETH) for speed
    sym = "ETHUSDT"
    samples = generate_training_samples(sym, test_month['train_start'], test_month['train_end'])
    model = RandomForestClassifier(
        n_estimators=80, max_depth=5, min_samples_leaf=15,
        random_state=42, class_weight="balanced")
    X = np.array([[s[c] for c in FEATURE_COLS] for s in samples])
    y = np.array([s["label"] for s in samples])
    model.fit(X, y)

    ai_per_symbol = {sym: model}
    threshold_per_symbol = {sym: 0.45}

    # Override CONFIG to just ETH
    original_symbols = CONFIG["symbols"]
    CONFIG["symbols"] = [sym]

    trades, stats = backtest_month(
        ai_per_symbol, threshold_per_symbol, risk_cfg,
        test_month['test_start'], test_month['test_end']
    )
    CONFIG["symbols"] = original_symbols

    # Analyze position sizing
    print(f"\n  Total trades: {len(trades)}")
    print(f"\n  Position sizing audit:")
    print(f"  {'#':>3} {'Risk %':>8} {'Risk $':>10} {'Wallet Before':>14} {'Wallet After':>14} {'Net P&L':>10} {'R Mult':>7}")
    print("-"*80)

    max_risk_pct = 0; min_risk_pct = 1
    wallet_growth = []
    for i, t in enumerate(trades):
        rp = t["risk_pct"]
        max_risk_pct = max(max_risk_pct, rp)
        min_risk_pct = min(min_risk_pct, rp)
        wallet_growth.append(t["wallet_after"])
        if i < 30 or i >= len(trades) - 10:  # first 30 + last 10
            print(f"  {i+1:>3} {rp*100:>7.2f}% {t['risk_usd']:>10.2f} "
                  f"{t['wallet_before']:>14.2f} {t['wallet_after']:>14.2f} "
                  f"{t['net_pnl']:>+10.2f} {t['R_multiple']:>7.2f}")
        elif i == 30:
            print(f"  ... ({len(trades)-40} more trades) ...")

    print(f"\n  Risk % range: {min_risk_pct*100:.2f}% to {max_risk_pct*100:.2f}%")
    print(f"  Configured range: {risk_cfg.anti_mart_min_risk*100:.2f}% to {risk_cfg.anti_mart_max_risk*100:.2f}%")
    print(f"  Within bounds: {'YES' if min_risk_pct >= risk_cfg.anti_mart_min_risk and max_risk_pct <= risk_cfg.anti_mart_max_risk else 'NO — BUG!'}")

    # Check for runaway compounding
    if len(wallet_growth) > 1:
        initial = INITIAL_WALLET
        final = wallet_growth[-1]
        growth_mult = final / initial
        print(f"\n  Wallet growth: ${initial:,.2f} → ${final:,.2f} ({growth_mult:.2f}x)")
        if growth_mult > 5:
            print(f"  ⚠ WARNING: Wallet grew {growth_mult:.1f}x in one month — possible runaway compounding")
        else:
            print(f"  ✓ Wallet growth is reasonable for one month")

    # Check consecutive win/loss streaks
    streaks = []
    cur_streak = 0; cur_type = None
    for t in trades:
        if t["net_pnl"] > 0:
            if cur_type == "win": cur_streak += 1
            else:
                if cur_type: streaks.append((cur_type, cur_streak))
                cur_type = "win"; cur_streak = 1
        else:
            if cur_type == "loss": cur_streak += 1
            else:
                if cur_type: streaks.append((cur_type, cur_streak))
                cur_type = "loss"; cur_streak = 1
    if cur_type: streaks.append((cur_type, cur_streak))

    max_win_streak = max((s for s in streaks if s[0] == "win"), key=lambda x: x[1], default=("win", 0))[1]
    max_loss_streak = max((s for s in streaks if s[0] == "loss"), key=lambda x: x[1], default=("loss", 0))[1]
    print(f"\n  Max consecutive wins: {max_win_streak}")
    print(f"  Max consecutive losses: {max_loss_streak}")
    print(f"  Consec loss throttle 1 (halve size): {risk_cfg.consec_loss_throttle_1}")
    print(f"  Consec loss throttle 2 (24h pause): {risk_cfg.consec_loss_throttle_2}")

    return trades, stats


def run_lookahead_smoke_test():
    """TEST 5: Smoke test — verify indicator values don't change when future data is added."""
    print("\n" + "="*100)
    print("TEST 5: LOOKAHEAD SMOKE TEST — Verify indicators don't use future data")
    print("="*100)

    sym = "BTCUSDT"
    df = load_data(sym)
    test_idx = 5000  # arbitrary index

    # Compute indicators on full data
    ind_full = get_indicators(df, sym)
    _IND_CACHE.clear()

    # Compute indicators on truncated data (only first 5001 bars)
    df_trunc = df.iloc[:test_idx+1]
    ind_trunc = get_indicators(df_trunc, sym)
    _IND_CACHE.clear()

    # Compare values at test_idx
    print(f"\n  Comparing indicator values at bar {test_idx}:")
    print(f"  {'Indicator':<15} {'Full data':>15} {'Truncated':>15} {'Match?':>8}")
    print("-"*55)

    all_match = True
    for key in ["rsi", "adx", "atr", "ema20", "ema50", "ema200", "slope", "ret_3", "ret_12"]:
        val_full = ind_full[key].iloc[test_idx]
        val_trunc = ind_trunc[key].iloc[test_idx]
        match = np.isclose(val_full, val_trunc, equal_nan=True)
        if not match: all_match = False
        print(f"  {key:<15} {val_full:>15.6f} {val_trunc:>15.6f} {'✓' if match else '✗ BUG!':>8}")

    # Cross-asset feature test
    print(f"\n  Cross-asset feature test:")
    t = df.index[test_idx]
    cross_full = get_cross_asset_features("BTCUSDT", t)
    # The cross-asset features should be the same regardless of future data
    print(f"  Time: {t}")
    print(f"  ETH_ret_3: {cross_full.get('ETH_ret_3', 0):.6f}")
    print(f"  ETH_rsi:   {cross_full.get('ETH_rsi', 0):.6f}")
    print(f"  ETH_adx:   {cross_full.get('ETH_adx', 0):.6f}")

    # Verify: the ETH value should come from the bar BEFORE t, not at or after t
    eth_df = load_data("ETHUSDT")
    eth_idx = eth_df.index.searchsorted(t, side="left") - 1
    eth_time = eth_df.index[eth_idx]
    print(f"  ETH bar used: index={eth_idx}, time={eth_time} (should be BEFORE {t})")
    print(f"  ETH bar is before signal time: {eth_time < t}")

    if all_match:
        print(f"\n  ✓ ALL INDICATORS MATCH — no lookahead from indicator computation")
    else:
        print(f"\n  ✗ LOOKAHEAD DETECTED — some indicators use future data!")

    return all_match


def main():
    print("="*100)
    print("AIv3_029 COMPREHENSIVE AUDIT — 5 tests")
    print("="*100)

    print("\nBUGS FOUND IN CODE AUDIT:")
    print("  Bug A: Trailing stop BROKEN — pos['best_price'] never updated (conservative)")
    print("  Bug B: Stop never moves to breakeven (same root cause as Bug A)")
    print("  Bug C: risk_amount uses pre-commission wallet (minor)")
    print("  Bug D: NO LOOKAHEAD FOUND in cross-asset features or indicators")

    # Test 5 first (fast)
    lookahead_ok = run_lookahead_smoke_test()

    # Test 4 (fast — one symbol, one month)
    pos_trades, pos_stats = run_position_sizing_audit()

    # Test 2 (medium — 6 months, no AI)
    no_ai_results = run_no_ai_baseline()

    # Test 1 (medium — 6 months, coin flip)
    coinflip_results = run_coinflip_control()

    # Test 3 (slow — fetches new data + trains + tests)
    try:
        oos_trades, oos_stats = run_out_of_sample()
    except Exception as e:
        print(f"\n  OOS test failed: {e}")
        oos_stats = {"n_trades": 0, "net": 0, "net_pct": 0, "win_rate": 0}

    # Summary
    print("\n" + "="*100)
    print("AUDIT SUMMARY")
    print("="*100)

    print(f"\n1. LOOKAHEAD SMOKE TEST: {'PASS' if lookahead_ok else 'FAIL'}")
    print(f"   Indicators computed on full vs truncated data match: {lookahead_ok}")

    print(f"\n2. NO-AI BASELINE (take every signal, 6 months):")
    n_prof_noai = sum(1 for r in no_ai_results if r['net'] > 0)
    avg_net_noai = np.mean([r['net'] for r in no_ai_results])
    avg_wr_noai = np.mean([r['win_rate'] for r in no_ai_results if r['n_trades'] > 0])
    print(f"   Profitable: {n_prof_noai}/6, avg net=${avg_net_noai:+,.2f}, avg WR={avg_wr_noai*100:.1f}%")

    print(f"\n3. COIN-FLIP CONTROL (random AI, 6 months):")
    n_prof_cf = sum(1 for r in coinflip_results if r['net'] > 0)
    avg_net_cf = np.mean([r['net'] for r in coinflip_results])
    avg_wr_cf = np.mean([r['win_rate'] for r in coinflip_results if r['n_trades'] > 0])
    print(f"   Profitable: {n_prof_cf}/6, avg net=${avg_net_cf:+,.2f}, avg WR={avg_wr_cf*100:.1f}%")

    print(f"\n4. OUT-OF-SAMPLE ASSETS (6 new assets, 1 month):")
    print(f"   Trades: {oos_stats['n_trades']}, WR={oos_stats.get('win_rate',0)*100:.1f}%, "
          f"net=${oos_stats.get('net',0):+,.2f} ({oos_stats.get('net_pct',0)*100:+.2f}%)")

    print(f"\n5. POSITION SIZING (1 month, ETH only):")
    print(f"   Trades: {pos_stats['n_trades']}, net=${pos_stats['net']:+,.2f}")

    print("\n" + "="*100)
    print("VERDICT")
    print("="*100)
    if avg_net_noai > 0 and n_prof_noai >= 4:
        print("  The edge is from MOMENTUM BARS + RISK MANAGEMENT, not the AI.")
        print("  No-AI baseline is also profitable — AI adds marginal value.")
    elif avg_net_noai > 0:
        print("  The edge is PARTIALLY from momentum bars + risk management.")
        print("  AI may add value but needs more testing.")
    else:
        print("  The edge is from the AI. Without AI, the system loses money.")

    if avg_net_cf > 0:
        print("  ⚠ COIN FLIP IS ALSO PROFITABLE — the edge is NOT from AI predictions.")
        print("  The profitability comes from the momentum bar generator + risk management.")
    else:
        print("  ✓ Coin flip is NOT profitable — the AI IS adding value.")

    if oos_stats.get('net', 0) > 0:
        print("  ✓ OUT-OF-SAMPLE ASSETS are profitable — edge generalizes.")
    else:
        print("  ✗ OUT-OF-SAMPLE ASSETS lost money — edge doesn't generalize.")


if __name__ == "__main__":
    main()
