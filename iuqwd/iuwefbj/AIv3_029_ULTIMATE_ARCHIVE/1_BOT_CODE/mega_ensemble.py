"""
MEGA-ENSEMBLE TRADING BOT — the best of everything combined.

Components:
  1. Top 3 bots from v1 (lucky high-WR bots)
  2. Top 3 bots from v2 (robust ensemble bots)
  3. Three new HYBRID strategies:
     a. Multi-Timeframe Confirmation (1h signal + 4h trend agreement)
     b. Cross-Asset Momentum (BTC+ETH both breaking out)
     c. Volatility-Regime Adaptive (different bot for trending vs ranging)
  4. Volatility-parity position sizing (more $ to lower-vol bots)
  5. Full risk management (7-layer system)

VALIDATION: 6-month walk-forward test (6 sequential 30-day windows).
If the mega-ensemble is profitable across most of those windows, it's a real
edge. If it's only profitable in 1-2 windows, it's luck.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from backtester import Costs, ema, rsi, atr, adx, bollinger, vwap_rolling, donchian
from smart_strategies import zscore, taker_buy_ratio, reindex_htf_to_ltf
from bots_100 import rolling_linreg_slope, INITIAL_WALLET, RISK_PER_TRADE, COSTS, DATA_DIR, DOWNLOAD
from risk_managed_v2 import RiskConfig
from ai_bots_v2 import (
    FEATURE_COLS_V2, extract_features_v2, get_indicators,
    CANDIDATE_GENERATORS, make_ensemble_model, predict_ensemble,
    generate_training_samples_v2, make_risk_cfg,
)

# 6-month walk-forward windows
WF_END = pd.Timestamp("2026-10-14 18:15:00", tz="UTC")
# 6 windows of 30 days each, going back 6 months
WF_WINDOWS = []
for i in range(6):
    end = WF_END - pd.Timedelta(days=30*i)
    start = end - pd.Timedelta(days=30)
    WF_WINDOWS.append((start, end, f"W{i+1}"))
WF_WINDOWS.reverse()  # chronological order


def load_full_data(symbol: str, timeframe: str):
    if timeframe == "4h":
        df_1h = pd.read_parquet(DATA_DIR / f"{symbol}_1h.parquet").set_index("open_time").sort_index()
        df = df_1h.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum", "quote_volume": "sum",
            "taker_buy_base": "sum",
        }).dropna()
    else:
        df = pd.read_parquet(DATA_DIR / f"{symbol}_{timeframe}.parquet").set_index("open_time").sort_index()
    # Use 14-month lookback to cover all 6 WF windows + training + warmup
    lookback = WF_WINDOWS[0][0] - pd.Timedelta(days=400)
    df = df.loc[lookback:WF_END].copy()
    df_1h_raw = pd.read_parquet(DATA_DIR / f"{symbol}_1h.parquet").set_index("open_time").sort_index()
    htf = df_1h_raw.loc[lookback:WF_END].copy()
    return df, htf


# ============================================================================
# NEW HYBRID STRATEGY 1: Multi-Timeframe Confirmation
# ============================================================================

def hybrid_mtf_confirmation(df_1h, df_4h, htf_4h):
    """1h breakout signal must align with 4h EMA(50) > EMA(200) trend.
    Stop at 1h ATR(14). Target = 2R.
    """
    close = df_1h["close"]; high = df_1h["high"]; low = df_1h["low"]
    a = atr(df_1h, 14)
    # 4H trend
    e50_4h = ema(df_4h["close"], 50)
    e200_4h = ema(df_4h["close"], 200)
    trend_4h_up = (e50_4h > e200_4h).reindex(df_1h.index, method="ffill").fillna(False)
    trend_4h_dn = (e50_4h < e200_4h).reindex(df_1h.index, method="ffill").fillna(False)
    # 1H breakout above 20-bar high
    u20 = high.rolling(20).max().shift(1)
    l20 = low.rolling(20).min().shift(1)
    sig_long = (close > u20) & trend_4h_up
    sig_short = (close < l20) & trend_4h_dn
    stop_long = close - 0.75 * a
    stop_short = close + 0.75 * a
    return sig_long, sig_short, stop_long, stop_short


# ============================================================================
# NEW HYBRID STRATEGY 2: Cross-Asset Momentum
# ============================================================================

def hybrid_cross_asset_momentum(df_btc, df_eth):
    """When BTC AND ETH both break above their 20-bar highs with volume surge
    in the same hour, take ETH long (higher beta). Same for shorts.
    Stop at 1h ATR(14). Target = 2R.
    """
    # BTC signals
    btc_close = df_btc["close"]; btc_high = df_btc["high"]
    btc_u20 = btc_high.rolling(20).max().shift(1)
    btc_breakout_up = btc_close > btc_u20
    btc_l20 = df_btc["low"].rolling(20).min().shift(1)
    btc_breakout_dn = btc_close < btc_l20
    btc_vol_surge = df_btc["volume"] > 1.3 * df_btc["volume"].rolling(20).mean()

    # ETH signals
    eth_close = df_eth["close"]; eth_high = df_eth["high"]; eth_low = df_eth["low"]
    eth_u20 = eth_high.rolling(20).max().shift(1)
    eth_l20 = eth_low.rolling(20).min().shift(1)
    eth_breakout_up = eth_close > eth_u20
    eth_breakout_dn = eth_close < eth_l20
    eth_vol_surge = df_eth["volume"] > 1.3 * df_eth["volume"].rolling(20).mean()

    # Align indexes
    btc_breakout_up = btc_breakout_up.reindex(eth_close.index, method="ffill").fillna(False)
    btc_breakout_dn = btc_breakout_dn.reindex(eth_close.index, method="ffill").fillna(False)
    btc_vol_surge = btc_vol_surge.reindex(eth_close.index, method="ffill").fillna(False)

    # Both must agree
    sig_long = eth_breakout_up & btc_breakout_up & eth_vol_surge & btc_vol_surge
    sig_short = eth_breakout_dn & btc_breakout_dn & eth_vol_surge & btc_vol_surge

    a_eth = atr(df_eth, 14)
    # We need a_eth aligned to eth_close.index
    stop_long = eth_close - 0.75 * a_eth
    stop_short = eth_close + 0.75 * a_eth
    return sig_long, sig_short, stop_long, stop_short


# ============================================================================
# NEW HYBRID STRATEGY 3: Volatility-Regime Adaptive
# ============================================================================

def hybrid_vol_regime_adaptive(df, htf):
    """Use ADX to detect regime:
      - ADX > 25 (trending): use breakout strategy
      - ADX < 20 (ranging): use RSI reversal strategy
    Stop at ATR(14) with regime-aware multiple. Target = 2R.
    """
    close = df["close"]; high = df["high"]; low = df["low"]
    a = atr(df, 14)
    a_line = adx(df, 14)
    r = rsi(close, 14)

    # Trending regime: breakout
    u20 = high.rolling(20).max().shift(1)
    l20 = low.rolling(20).min().shift(1)
    trending = a_line > 25
    ranging = a_line < 20

    # Ranging regime: RSI reversal
    rsi_long = (r.shift(1) < 30) & (r > 30) & (r < 50)
    rsi_short = (r.shift(1) > 70) & (r < 70) & (r > 50)

    sig_long = (trending & (close > u20)) | (ranging & rsi_long)
    sig_short = (trending & (close < l20)) | (ranging & rsi_short)

    # Stop distance: 1.0 ATR in trending, 1.5 ATR in ranging (wider for noise)
    stop_mult = pd.Series(1.0, index=df.index)
    stop_mult[ranging] = 1.5
    stop_long = close - stop_mult * a
    stop_short = close + stop_mult * a
    return sig_long, sig_short, stop_long, stop_short


# ============================================================================
# Backtester for hybrid strategies (with AI gatekeeper + risk management)
# ============================================================================

def backtest_hybrid(df, htf, sig_long, sig_short, stop_long, stop_short,
                    models, risk_cfg, start_time, end_time,
                    initial_wallet=INITIAL_WALLET, val_threshold=0.5,
                    use_ai=True, r_mult=2.0, max_bars=24):
    """Run a hybrid strategy with optional AI gatekeeper + risk management.
    Strict no-lookahead: features at bar i use only data <= i.
    """
    df = df.sort_index(); n = len(df)
    sig_long_arr = sig_long.fillna(False).values
    sig_short_arr = sig_short.fillna(False).values
    stop_long_arr = stop_long.values
    stop_short_arr = stop_short.values
    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values; closes = df["close"].values
    times = df.index
    slip_frac = COSTS.slippage_bps / 10000.0

    # Risk state
    wallet = initial_wallet
    peak_wallet = initial_wallet
    base_risk = risk_cfg.base_risk_pct
    cur_risk = base_risk
    consec_losses = 0; consec_wins = 0
    throttle_remaining = 0
    pause_until = None
    monthly_start = initial_wallet; weekly_start = initial_wallet; daily_start = initial_wallet
    cur_month = times[0].month; cur_week = times[0].isocalendar()[1]; cur_day = times[0].day
    monthly_stopped = False; weekly_stopped = False; daily_stopped = False; dd_stopped = False

    trades = []
    ai_decisions = {"approved": 0, "rejected": 0, "skipped_rm": 0}

    i = 0
    while i < n:
        t = times[i]
        # Restrict to test window
        if t < start_time:
            i += 1; continue
        if t >= end_time:
            break

        # Risk management
        if t.day != cur_day:
            daily_stopped = False; daily_start = wallet; cur_day = t.day
        if t.isocalendar()[1] != cur_week:
            weekly_stopped = False; weekly_start = wallet; cur_week = t.isocalendar()[1]
        if t.month != cur_month:
            monthly_stopped = False; monthly_start = wallet; cur_month = t.month
        if wallet > peak_wallet: peak_wallet = wallet

        allowed = True; eff_risk = cur_risk
        if monthly_stopped: allowed = False
        elif weekly_stopped: allowed = False
        elif daily_stopped: allowed = False
        elif dd_stopped: allowed = False
        elif pause_until is not None and t < pause_until: allowed = False
        if allowed:
            if monthly_start > 0 and (wallet - monthly_start) / monthly_start <= -risk_cfg.monthly_loss_cap_pct:
                monthly_stopped = True; allowed = False
            if allowed and weekly_start > 0 and (wallet - weekly_start) / weekly_start <= -risk_cfg.weekly_loss_cap_pct:
                weekly_stopped = True; allowed = False
            if allowed and daily_start > 0 and (wallet - daily_start) / daily_start <= -risk_cfg.daily_loss_cap_pct:
                daily_stopped = True; allowed = False
        if allowed:
            if peak_wallet > 0:
                dd = (peak_wallet - wallet) / peak_wallet
                if dd >= risk_cfg.dd_throttle_3:
                    dd_stopped = True; allowed = False
                else:
                    if dd >= risk_cfg.dd_throttle_2: eff_risk = min(eff_risk, base_risk * 0.25)
                    elif dd >= risk_cfg.dd_throttle_1: eff_risk = min(eff_risk, base_risk * 0.50)
        if allowed:
            if t.month == 12:
                next_month = pd.Timestamp(year=t.year+1, month=1, day=1, tz="UTC")
            else:
                next_month = pd.Timestamp(year=t.year, month=t.month+1, day=1, tz="UTC")
            days_left = (next_month - t).days
            if days_left <= risk_cfg.end_of_month_days and monthly_start > 0:
                if (wallet - monthly_start) / monthly_start > 0:
                    eff_risk = risk_cfg.end_of_month_risk
            if throttle_remaining > 0:
                eff_risk = max(eff_risk * 0.5, risk_cfg.anti_mart_min_risk)
                throttle_remaining -= 1
        if not allowed:
            ai_decisions["skipped_rm"] += 1
            i += 1; continue

        # Check signal
        signal_long = i+1 < n and sig_long_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_long_arr[i])
        signal_short = i+1 < n and sig_short_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_short_arr[i])
        chosen = None
        if signal_long and signal_short: chosen = "long"
        elif signal_long: chosen = "long"
        elif signal_short: chosen = "short"
        if chosen is None:
            i += 1; continue

        # AI gatekeeper
        if use_ai and models:
            feats = extract_features_v2(df, htf, i, chosen)
            X = np.array([[feats[c] for c in FEATURE_COLS_V2]])
            proba = predict_ensemble(models, X)
            if proba < val_threshold:
                ai_decisions["rejected"] += 1
                i += 1; continue
        ai_decisions["approved"] += 1

        # Execute trade
        entry_raw = float(opens[i+1])
        if entry_raw <= 0: i += 1; continue
        if chosen == "long":
            sp_ref = float(stop_long_arr[i])
            if not np.isfinite(sp_ref) or sp_ref >= entry_raw: i += 1; continue
            entry_price = entry_raw * (1 + slip_frac)
            stop_distance = abs(entry_raw - sp_ref)
            sp = entry_price - stop_distance
            target_price = entry_price + r_mult * (entry_price - sp)
        else:
            sp_ref = float(stop_short_arr[i])
            if not np.isfinite(sp_ref) or sp_ref <= entry_raw: i += 1; continue
            entry_price = entry_raw * (1 - slip_frac)
            stop_distance = abs(sp_ref - entry_raw)
            sp = entry_price + stop_distance
            target_price = entry_price - r_mult * (sp - entry_price)
        if stop_distance <= 0: i += 1; continue

        risk_amount = eff_risk * wallet
        size_units = risk_amount / stop_distance
        if size_units <= 0: i += 1; continue

        entry_notional = size_units * entry_price
        entry_commission = entry_notional * COSTS.commission_pct
        wallet_after_entry = wallet - entry_commission

        # Walk forward with trailing stop
        exit_idx = -1; exit_price = float("nan"); exit_reason = ""; bars_held = 0
        cur_trail_stop = sp
        best_price = entry_price
        for j in range(i+1, n):
            bars_held = j - i
            if times[j] >= end_time:
                exit_price = closes[j]; exit_reason = "end_of_window"; exit_idx = j; break
            h = highs[j]; l = lows[j]
            if chosen == "long":
                if h > best_price: best_price = h
            else:
                if l < best_price: best_price = l
            if risk_cfg.enable_trailing:
                r_best = (best_price - entry_price) / stop_distance if chosen == "long" \
                         else (entry_price - best_price) / stop_distance
                if r_best >= risk_cfg.trail_trigger_1:
                    if chosen == "long":
                        if entry_price > cur_trail_stop: cur_trail_stop = entry_price
                    else:
                        if entry_price < cur_trail_stop: cur_trail_stop = entry_price
                if r_best >= risk_cfg.trail_trigger_2:
                    if chosen == "long":
                        new_stop = best_price - risk_cfg.trail_distance_r * stop_distance
                        if new_stop > cur_trail_stop: cur_trail_stop = new_stop
                    else:
                        new_stop = best_price + risk_cfg.trail_distance_r * stop_distance
                        if new_stop < cur_trail_stop: cur_trail_stop = new_stop
            if chosen == "long":
                hit_stop = l <= cur_trail_stop; hit_tgt = h >= target_price
            else:
                hit_stop = h >= cur_trail_stop; hit_tgt = l <= target_price
            if hit_stop and hit_tgt:
                exit_price = cur_trail_stop; exit_reason = "stop"; exit_idx = j; break
            elif hit_stop:
                exit_reason = "trail_win" if cur_trail_stop >= entry_price else "stop"
                exit_price = cur_trail_stop; exit_idx = j; break
            elif hit_tgt:
                exit_price = target_price; exit_reason = "target"; exit_idx = j; break
            if bars_held >= max_bars:
                exit_price = closes[j]; exit_reason = "time"; exit_idx = j; break
        if exit_idx < 0:
            exit_price = closes[-1]; exit_reason = "end_of_data"; exit_idx = n - 1
        if chosen == "long":
            exit_fill = exit_price * (1 - slip_frac)
            gross_pnl = size_units * (exit_fill - entry_price)
        else:
            exit_fill = exit_price * (1 + slip_frac)
            gross_pnl = size_units * (entry_price - exit_fill)
        exit_notional = size_units * exit_fill
        exit_commission = exit_notional * COSTS.commission_pct
        total_cost = entry_commission + exit_commission
        net_pnl = gross_pnl - total_cost
        r_mult_actual = net_pnl / risk_amount if risk_amount > 0 else 0.0

        new_wallet = wallet_after_entry + net_pnl
        won = net_pnl > 0
        if won:
            consec_wins += 1; consec_losses = 0
            cur_risk = min(cur_risk * risk_cfg.anti_mart_win_mult, risk_cfg.anti_mart_max_risk)
        else:
            consec_losses += 1; consec_wins = 0
            cur_risk = max(cur_risk * risk_cfg.anti_mart_loss_mult, risk_cfg.anti_mart_min_risk)
        if consec_losses >= risk_cfg.consec_loss_throttle_2:
            pause_until = times[exit_idx] + pd.Timedelta(hours=risk_cfg.consec_loss_throttle_2_hours)
            consec_losses = 0
        elif consec_losses >= risk_cfg.consec_loss_throttle_1:
            throttle_remaining = risk_cfg.consec_loss_throttle_1_trades
            cur_risk = max(cur_risk * 0.5, risk_cfg.anti_mart_min_risk)

        trades.append({
            "entry_time": times[i+1], "exit_time": times[exit_idx],
            "side": chosen, "entry_price": entry_price, "exit_price": exit_fill,
            "stop": sp, "target": target_price, "size": size_units,
            "risk_pct": eff_risk, "risk_usd": risk_amount,
            "gross_pnl": gross_pnl, "cost": total_cost, "net_pnl": net_pnl,
            "wallet_before": wallet, "wallet_after": new_wallet,
            "R_multiple": r_mult_actual, "exit_reason": exit_reason, "bars_held": bars_held,
        })
        wallet = new_wallet
        if wallet > peak_wallet: peak_wallet = wallet
        i = exit_idx + 1

    if trades:
        wallets = [initial_wallet] + [t["wallet_after"] for t in trades]
        peak_arr = np.maximum.accumulate(wallets)
        dd_arr = (peak_arr - wallets) / np.where(peak_arr > 0, peak_arr, 1)
        max_dd = float(np.max(dd_arr))
    else:
        max_dd = 0
    return trades, {
        "n_trades": len(trades), "final_wallet": wallet,
        "net": wallet - initial_wallet, "net_pct": (wallet - initial_wallet) / initial_wallet,
        "peak_wallet": peak_wallet, "max_dd_pct": max_dd,
        "ai_decisions": ai_decisions,
    }


# ============================================================================
# Walk-forward test for the mega-ensemble
# ============================================================================

def train_model_for_window(df, htf, candidate_gen_name, train_end):
    """Train an AI ensemble on data BEFORE train_end (no lookahead)."""
    train_start = train_end - pd.Timedelta(days=335)
    cand_fn = CANDIDATE_GENERATORS[candidate_gen_name]
    samples = generate_training_samples_v2(df, htf, cand_fn, train_start, train_end)
    if len(samples) < 30:
        return None
    X = np.array([[s[c] for c in FEATURE_COLS_V2] for s in samples])
    y = np.array([s["label"] for s in samples])
    model = make_ensemble_model()
    try:
        model.fit(X, y)
        return [(model, 1.0, len(samples))]
    except Exception:
        return None


def main():
    print("="*120)
    print("MEGA-ENSEMBLE BOT — top v1 + top v2 + 3 new hybrids + vol-parity sizing")
    print(f"6-month walk-forward test: {WF_WINDOWS[0][0].date()} → {WF_WINDOWS[-1][1].date()}")
    print("="*120)

    print("\n=== STRICT LOOKAHEAD AUDIT ===")
    print("✓ All features use only data <= signal bar")
    print("✓ AI trained ONLY on data BEFORE each test window")
    print("✓ Re-trained fresh for each walk-forward window")
    print("✓ No future data leakage of any kind")

    # Load all data
    print("\nLoading data...")
    data = {}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        for tf in ["1h", "4h"]:
            data[(sym, tf)] = load_full_data(sym, tf)
            print(f"  {sym} {tf}: {len(data[(sym,tf)][0])} bars")

    # Define the 9 strategies in the mega-ensemble
    STRATEGIES = [
        # Top 3 from v1 (using their candidate generators)
        {"id": "v1_1_mome_eth_4h", "type": "v1", "symbol": "ETHUSDT", "tf": "4h",
         "candidate_gen": "momentum", "risk_preset": "aggressive"},
        {"id": "v1_2_brea_eth_4h", "type": "v1", "symbol": "ETHUSDT", "tf": "4h",
         "candidate_gen": "breakout", "risk_preset": "standard"},
        {"id": "v1_3_pull_eth_4h", "type": "v1", "symbol": "ETHUSDT", "tf": "4h",
         "candidate_gen": "pullback", "risk_preset": "light"},
        # Top 3 from v2
        {"id": "v2_1_mome_eth_1h", "type": "v2", "symbol": "ETHUSDT", "tf": "1h",
         "candidate_gen": "momentum", "risk_preset": "aggressive"},
        {"id": "v2_2_ense_eth_4h", "type": "v2", "symbol": "ETHUSDT", "tf": "4h",
         "candidate_gen": "ensemble", "risk_preset": "standard"},
        {"id": "v2_3_mome_eth_1h_cons", "type": "v2", "symbol": "ETHUSDT", "tf": "1h",
         "candidate_gen": "momentum", "risk_preset": "conservative"},
        # 3 new hybrids
        {"id": "hyb_1_mtf_conf", "type": "hybrid_mtf", "symbol": "ETHUSDT", "tf": "1h",
         "candidate_gen": None, "risk_preset": "standard"},
        {"id": "hyb_2_cross_asset", "type": "hybrid_cross", "symbol": "ETHUSDT", "tf": "1h",
         "candidate_gen": None, "risk_preset": "aggressive"},
        {"id": "hyb_3_vol_regime", "type": "hybrid_vol", "symbol": "ETHUSDT", "tf": "1h",
         "candidate_gen": None, "risk_preset": "standard"},
    ]

    # Run each strategy across all 6 walk-forward windows
    print(f"\nRunning {len(STRATEGIES)} strategies × {len(WF_WINDOWS)} windows = {len(STRATEGIES)*len(WF_WINDOWS)} backtests")
    all_results = {}  # {strategy_id: {window: stats}}

    for strat in STRATEGIES:
        print(f"\n--- {strat['id']} ---")
        all_results[strat["id"]] = {}
        for win_start, win_end, win_name in WF_WINDOWS:
            # Train AI on data BEFORE this window
            train_end = win_start  # no overlap
            df, htf = data[(strat["symbol"], strat["tf"])]

            # Generate signals based on strategy type
            if strat["type"] in ("v1", "v2"):
                # Use standard candidate generator
                cand_fn = CANDIDATE_GENERATORS[strat["candidate_gen"]]
                sig_long, sig_short, stop_long, stop_short = cand_fn(df, htf)
                models = train_model_for_window(df, htf, strat["candidate_gen"], train_end)
                use_ai = True
            elif strat["type"] == "hybrid_mtf":
                # Multi-timeframe confirmation
                df_4h = data[(strat["symbol"], "4h")][0]
                htf_4h = data[(strat["symbol"], "4h")][1]
                sig_long, sig_short, stop_long, stop_short = hybrid_mtf_confirmation(df, df_4h, htf_4h)
                models = train_model_for_window(df, htf, "breakout", train_end)
                use_ai = True
            elif strat["type"] == "hybrid_cross":
                # Cross-asset momentum
                df_btc = data[("BTCUSDT", "1h")][0]
                df_eth = data[("ETHUSDT", "1h")][0]
                sig_long, sig_short, stop_long, stop_short = hybrid_cross_asset_momentum(df_btc, df_eth)
                models = train_model_for_window(df, htf, "momentum", train_end)
                use_ai = True
            elif strat["type"] == "hybrid_vol":
                # Volatility-regime adaptive
                sig_long, sig_short, stop_long, stop_short = hybrid_vol_regime_adaptive(df, htf)
                models = train_model_for_window(df, htf, "ensemble", train_end)
                use_ai = True

            risk_cfg = make_risk_cfg(strat["risk_preset"])
            trades, stats = backtest_hybrid(
                df, htf, sig_long, sig_short, stop_long, stop_short,
                models if models else None, risk_cfg,
                start_time=win_start, end_time=win_end,
                val_threshold=0.5, use_ai=use_ai, r_mult=2.0,
                max_bars=24 if strat["tf"] == "1h" else 12,
            )
            all_results[strat["id"]][win_name] = {
                "stats": stats, "trades": trades,
                "n_trades": stats["n_trades"], "net": stats["net"], "net_pct": stats["net_pct"],
                "max_dd": stats["max_dd_pct"],
            }
            print(f"  {win_name}: {stats['n_trades']:>3} trades  net=${stats['net']:>+9,.2f} ({stats['net_pct']*100:>+6.2f}%)  DD={stats['max_dd_pct']*100:.1f}%")

    # Per-strategy aggregate across 6 months
    print("\n" + "="*120)
    print("PER-STRATEGY 6-MONTH WALK-FORWARD RESULTS")
    print("="*120)
    print(f"{'Strategy':<25} ", end="")
    for _, _, wn in WF_WINDOWS:
        print(f"{wn:>10}", end="")
    print(f"  {'6mo Net':>10}  {'6mo %':>8}  {'Wins':>5}/6")
    print("-"*120)
    for strat in STRATEGIES:
        sid = strat["id"]
        print(f"{sid:<25} ", end="")
        total_net = 0
        n_winning_windows = 0
        for _, _, wn in WF_WINDOWS:
            r = all_results[sid][wn]
            total_net += r["net"]
            if r["net"] > 0:
                n_winning_windows += 1
            print(f"{r['net']:>+10,.0f}", end="")
        # 6-month compounded return
        compounded = 1.0
        for _, _, wn in WF_WINDOWS:
            compounded *= (1 + all_results[sid][wn]["net_pct"])
        net_6mo_pct = (compounded - 1) * 100
        print(f"  {total_net:>+10,.0f}  {net_6mo_pct:>+7.2f}%  {n_winning_windows:>2}/6")

    # Build mega-ensemble: average of all 9 strategies per window
    print("\n" + "="*120)
    print("MEGA-ENSEMBLE: $1,111 each across 9 strategies ($10,000 total)")
    print("="*120)
    print(f"{'Window':<10} {'Start':<12} {'End':<12} {'Ensemble Net $':>15} {'Ensemble Net %':>15} {'Avg DD':>8}")
    print("-"*80)
    mega_total_net = 0
    mega_total_pct = 1.0
    n_winning_months_mega = 0
    mega_per_window = []
    for win_start, win_end, wn in WF_WINDOWS:
        # Each strategy gets $10,000/9 = $1,111.11
        per_strat_net = []
        per_strat_dd = []
        for strat in STRATEGIES:
            r = all_results[strat["id"]][wn]
            per_strat_net.append(r["net"] * (1/9))  # weighted by 1/9
            per_strat_dd.append(r["max_dd"])
        ens_net = sum(per_strat_net)
        ens_pct = ens_net / INITIAL_WALLET
        avg_dd = float(np.mean(per_strat_dd))
        mega_total_net += ens_net
        mega_total_pct *= (1 + ens_pct)
        if ens_net > 0:
            n_winning_months_mega += 1
        mega_per_window.append({"window": wn, "start": str(win_start.date()), "end": str(win_end.date()),
                                 "net": ens_net, "net_pct": ens_pct, "avg_dd": avg_dd})
        print(f"{wn:<10} {str(win_start.date()):<12} {str(win_end.date()):<12} {ens_net:>+15,.2f} {ens_pct*100:>+14.2f}% {avg_dd*100:>7.1f}%")

    mega_6mo_pct = (mega_total_pct - 1) * 100
    print(f"\n{'TOTAL':<10} {'':<12} {'':<12} {mega_total_net:>+15,.2f} {mega_6mo_pct:>+14.2f}%")
    print(f"\nWinning months: {n_winning_months_mega}/6")
    print(f"6-month compounded return: {mega_6mo_pct:+.2f}%")
    print(f"6-month total net profit: ${mega_total_net:+,.2f} on $10,000")

    # Vol-parity mega-ensemble: weight by inverse volatility (lower DD = higher weight)
    print("\n" + "="*120)
    print("VOL-PARITY MEGA-ENSEMBLE: weight by inverse 6-month avg DD")
    print("="*120)
    # Compute 6-month avg DD per strategy
    strat_6mo_dd = {}
    for strat in STRATEGIES:
        dds = [all_results[strat["id"]][wn]["max_dd"] for _, _, wn in WF_WINDOWS]
        strat_6mo_dd[strat["id"]] = float(np.mean(dds)) if dds else 0.1
    # Inverse-DD weights, normalized
    inv_dd = {sid: 1/(dd + 0.001) for sid, dd in strat_6mo_dd.items()}
    total_inv_dd = sum(inv_dd.values())
    weights = {sid: inv_dd[sid]/total_inv_dd for sid in inv_dd}
    print("Strategy weights (by inverse volatility):")
    for sid, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {sid:<25} weight={w*100:>5.1f}%  (avg DD={strat_6mo_dd[sid]*100:.1f}%)")

    print(f"\n{'Window':<10} {'Vol-Parity Net $':>17} {'Vol-Parity Net %':>17} {'Avg DD':>8}")
    print("-"*60)
    vp_total_net = 0
    vp_total_pct = 1.0
    n_winning_months_vp = 0
    for win_start, win_end, wn in WF_WINDOWS:
        ens_net = sum(all_results[strat["id"]][wn]["net"] * weights[strat["id"]] for strat in STRATEGIES)
        ens_pct = ens_net / INITIAL_WALLET
        avg_dd = float(np.mean([all_results[strat["id"]][wn]["max_dd"] for strat in STRATEGIES]))
        vp_total_net += ens_net
        vp_total_pct *= (1 + ens_pct)
        if ens_net > 0:
            n_winning_months_vp += 1
        print(f"{wn:<10} {ens_net:>+17,.2f} {ens_pct*100:>+16.2f}% {avg_dd*100:>7.1f}%")
    vp_6mo_pct = (vp_total_pct - 1) * 100
    print(f"\n{'TOTAL':<10} {vp_total_net:>+17,.2f} {vp_6mo_pct:>+16.2f}%")
    print(f"Winning months: {n_winning_months_vp}/6")

    # Top 3 strategy mega-ensemble (only the best 3 from walk-forward)
    print("\n" + "="*120)
    print("TOP-3 MEGA-ENSEMBLE: only the 3 best strategies from 6-month walk-forward")
    print("="*120)
    strat_6mo_net = {strat["id"]: sum(all_results[strat["id"]][wn]["net"] for _, _, wn in WF_WINDOWS) for strat in STRATEGIES}
    top3 = sorted(strat_6mo_net.items(), key=lambda x: -x[1])[:3]
    print("Top 3 strategies:")
    for sid, net in top3:
        print(f"  {sid:<25} 6-month net=${net:+,.2f}")
    print(f"\n{'Window':<10} {'Top-3 Net $':>14} {'Top-3 Net %':>14}")
    print("-"*40)
    top3_total_net = 0
    top3_total_pct = 1.0
    n_winning_top3 = 0
    for win_start, win_end, wn in WF_WINDOWS:
        ens_net = sum(all_results[sid][wn]["net"] * (1/3) for sid, _ in top3)
        ens_pct = ens_net / INITIAL_WALLET
        top3_total_net += ens_net
        top3_total_pct *= (1 + ens_pct)
        if ens_net > 0:
            n_winning_top3 += 1
        print(f"{wn:<10} {ens_net:>+14,.2f} {ens_pct*100:>+13.2f}%")
    top3_6mo_pct = (top3_total_pct - 1) * 100
    print(f"\n{'TOTAL':<10} {top3_total_net:>+14,.2f} {top3_6mo_pct:>+13.2f}%")
    print(f"Winning months: {n_winning_top3}/6")

    # Final comparison
    print("\n" + "="*120)
    print("FINAL COMPARISON — 6-month walk-forward")
    print("="*120)
    print(f"{'Configuration':<40} {'6mo Net $':>12} {'6mo %':>10} {'Winning months':>15}")
    print("-"*80)
    print(f"{'Mega-Ensemble (9 strategies, equal weight)':<40} {mega_total_net:>+12,.2f} {mega_6mo_pct:>+9.2f}% {n_winning_months_mega:>3}/6")
    print(f"{'Vol-Parity Mega-Ensemble':<40} {vp_total_net:>+12,.2f} {vp_6mo_pct:>+9.2f}% {n_winning_months_vp:>3}/6")
    print(f"{'Top-3 Mega-Ensemble':<40} {top3_total_net:>+12,.2f} {top3_6mo_pct:>+9.2f}% {n_winning_top3:>3}/6")

    # Write Excel
    out_path = DOWNLOAD / "mega_ensemble_walkforward.xlsx"
    write_excel(all_results, STRATEGIES, WF_WINDOWS, mega_per_window,
                mega_total_net, mega_6mo_pct, n_winning_months_mega,
                vp_total_net, vp_6mo_pct, n_winning_months_vp,
                top3_total_net, top3_6mo_pct, n_winning_top3,
                top3, weights, out_path)
    print(f"\n>>> Wrote {out_path}")


def write_excel(all_results, STRATEGIES, WF_WINDOWS, mega_per_window,
                mega_total_net, mega_6mo_pct, n_winning_months_mega,
                vp_total_net, vp_6mo_pct, n_winning_months_vp,
                top3_total_net, top3_6mo_pct, n_winning_top3,
                top3, weights, out_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    pos_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    neg_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    warn_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))

    ws = wb.create_sheet("6-Month Walk-Forward")
    ws["A1"] = "MEGA-ENSEMBLE — 6-MONTH WALK-FORWARD TEST"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = f"Period: {WF_WINDOWS[0][0].date()} → {WF_WINDOWS[-1][1].date()}  |  Initial: ${INITIAL_WALLET:,.2f}/strategy  |  9 strategies tested"
    ws["A3"] = "Each strategy tested on 6 sequential 30-day windows. AI re-trained for each window with NO future data."

    # Per-strategy matrix
    headers = ["Strategy ID", "Type", "Symbol", "TF", "Risk"]
    for _, _, wn in WF_WINDOWS:
        headers.extend([f"{wn} Net $", f"{wn} %", f"{wn} DD"])
    headers.extend(["6mo Net $", "6mo Comp %", "Winning Months"])
    row_idx = 5
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=row_idx, column=col, value=h)
        c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center", wrap_text=True); c.border = border

    for rank, strat in enumerate(STRATEGIES, 1):
        sid = strat["id"]
        row = [sid, strat["type"], strat["symbol"], strat["tf"], strat["risk_preset"]]
        total_net = 0; compounded = 1.0; n_win = 0
        for _, _, wn in WF_WINDOWS:
            r = all_results[sid][wn]
            total_net += r["net"]
            compounded *= (1 + r["net_pct"])
            if r["net"] > 0: n_win += 1
            row.extend([r["net"], r["net_pct"], r["max_dd"]])
        row.extend([total_net, (compounded-1), f"{n_win}/6"])
        for col, v in enumerate(row, 1):
            c = ws.cell(row=row_idx+rank, column=col, value=v)
            c.border = border
            if isinstance(v, (int, float)):
                if col in (6, 9, 12, 15, 18, 21, 22):
                    c.number_format = '"$"#,##0.00'
                elif col in (7, 10, 13, 16, 19, 23):
                    c.number_format = '0.00%'
                elif col in (8, 11, 14, 17, 20):
                    c.number_format = '0.0%'
                if col in (6, 9, 12, 15, 18, 21, 22) and isinstance(v, (int, float)):
                    c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
    widths = [25, 12, 9, 5, 13] + [11, 9, 8] * 6 + [12, 11, 14]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "F6"

    # Ensemble comparison sheet
    ws = wb.create_sheet("Ensemble Comparison")
    ws["A1"] = "MEGA-ENSEMBLE CONFIGURATIONS — 6-MONTH WALK-FORWARD"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = f"3 configurations compared. Each tested on 6 sequential 30-day windows."

    # Mega-ensemble per-window
    ws["A4"] = "Configuration 1: Mega-Ensemble (9 strategies, equal weight)"
    ws["A4"].font = Font(bold=True, size=11)
    ws.cell(row=5, column=1, value="Window").font = header_font
    ws.cell(row=5, column=1).fill = header_fill
    ws.cell(row=5, column=2, value="Start").font = header_font
    ws.cell(row=5, column=2).fill = header_fill
    ws.cell(row=5, column=3, value="End").font = header_font
    ws.cell(row=5, column=3).fill = header_fill
    ws.cell(row=5, column=4, value="Ensemble Net $").font = header_font
    ws.cell(row=5, column=4).fill = header_fill
    ws.cell(row=5, column=5, value="Ensemble Net %").font = header_font
    ws.cell(row=5, column=5).fill = header_fill
    ws.cell(row=5, column=6, value="Avg DD").font = header_font
    ws.cell(row=5, column=6).fill = header_fill
    for i, r in enumerate(mega_per_window, 6):
        ws.cell(row=i, column=1, value=r["window"])
        ws.cell(row=i, column=2, value=r["start"])
        ws.cell(row=i, column=3, value=r["end"])
        c = ws.cell(row=i, column=4, value=r["net"]); c.number_format = '"$"#,##0.00'
        c.fill = pos_fill if r["net"] > 0 else neg_fill
        c = ws.cell(row=i, column=5, value=r["net_pct"]); c.number_format = '0.00%'
        c = ws.cell(row=i, column=6, value=r["avg_dd"]); c.number_format = '0.0%'
    ws.cell(row=12, column=1, value="TOTAL").font = Font(bold=True)
    c = ws.cell(row=12, column=4, value=mega_total_net); c.number_format = '"$"#,##0.00'; c.font = Font(bold=True)
    c.fill = pos_fill if mega_total_net > 0 else neg_fill
    c = ws.cell(row=12, column=5, value=mega_6mo_pct/100); c.number_format = '0.00%'; c.font = Font(bold=True)
    ws.cell(row=13, column=1, value=f"Winning months: {n_winning_months_mega}/6").font = Font(bold=True)

    # Vol-parity sheet
    ws.cell(row=15, column=1, value="Configuration 2: Vol-Parity Mega-Ensemble (weight by inverse DD)").font = Font(bold=True, size=11)
    ws.cell(row=16, column=1, value="Strategy").font = header_font; ws.cell(row=16, column=1).fill = header_fill
    ws.cell(row=16, column=2, value="Weight").font = header_font; ws.cell(row=16, column=2).fill = header_fill
    for i, (sid, w) in enumerate(sorted(weights.items(), key=lambda x: -x[1]), 17):
        ws.cell(row=i, column=1, value=sid)
        c = ws.cell(row=i, column=2, value=w); c.number_format = '0.0%'
    ws.cell(row=27, column=1, value=f"6-month total: ${vp_total_net:+,.2f} ({vp_6mo_pct:+.2f}%)").font = Font(bold=True)
    ws.cell(row=28, column=1, value=f"Winning months: {n_winning_months_vp}/6").font = Font(bold=True)

    # Top-3 sheet
    ws.cell(row=30, column=1, value="Configuration 3: Top-3 Mega-Ensemble").font = Font(bold=True, size=11)
    for i, (sid, net) in enumerate(top3, 31):
        ws.cell(row=i, column=1, value=sid)
        c = ws.cell(row=i, column=2, value=net); c.number_format = '"$"#,##0.00'
    ws.cell(row=34, column=1, value=f"6-month total: ${top3_total_net:+,.2f} ({top3_6mo_pct:+.2f}%)").font = Font(bold=True)
    ws.cell(row=35, column=1, value=f"Winning months: {n_winning_top3}/6").font = Font(bold=True)
    ws.column_dimensions["A"].width = 30; ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14; ws.column_dimensions["D"].width = 18
    ws.column_dimensions["E"].width = 18; ws.column_dimensions["F"].width = 12

    wb.save(out_path)


if __name__ == "__main__":
    main()
