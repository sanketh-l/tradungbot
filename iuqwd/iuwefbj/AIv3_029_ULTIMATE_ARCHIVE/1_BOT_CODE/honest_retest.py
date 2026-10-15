"""
HONEST RE-TEST — Fix trailing stop bug, run 3 versions side by side.

BUG FIX: In the original code, pos["best_price"] was never updated — it
always stayed at entry_price. The trailing stop logic used a local variable
that reset every bar. This means trailing stops NEVER WORKED.

FIX: Update pos["best_price"] and pos["cur_trail_stop"] each bar so they
persist across bars.

3 VERSIONS TESTED (all with bug fix):
  A. WITH AI (original AIv3_029)
  B. NO AI (take every candidate — proven baseline)
  C. COIN FLIP (random 50/50 — control)

If all 3 produce similar results, the edge is from momentum + risk management.
If A >> B >> C, the AI adds real value.

Test: 6 months (for speed, since 36 months takes 7+ minutes and keeps dying)
"""
import sys, time, json, gc, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier

from backtester import Costs, ema, rsi, atr, adx, bollinger
from smart_strategies import zscore, taker_buy_ratio
from bots_100 import rolling_linreg_slope, INITIAL_WALLET, COSTS, DATA_DIR, DOWNLOAD
from risk_managed_v2 import RiskConfig
import walkforward_3year as wf
from walkforward_3year import (
    CONFIG, FEATURE_COLS, REF_SYMBOLS,
    load_data, get_indicators, get_cross_asset_features,
    extract_features, cand_momentum, generate_training_samples,
    make_risk_cfg, generate_test_months,
)
_IND_CACHE = wf._IND_CACHE

# Use last 6 months for speed
TEST_MONTHS = generate_test_months()[-6:]


# ============================================================================
# FIXED BACKTESTER — trailing stop bug corrected
# ============================================================================

def backtest_month_FIXED(ai_per_symbol, threshold_per_symbol, risk_cfg,
                          test_start, test_end, initial_wallet=INITIAL_WALLET):
    """Backtest with FIXED trailing stop logic.

    KEY FIX: pos["best_price"] and pos["cur_trail_stop"] are now updated
    every bar and persist across bars. Previously they were local variables
    that reset each bar, so trailing stops never worked.
    """
    symbols = CONFIG["symbols"]
    max_concurrent = CONFIG["max_concurrent"]
    max_same_direction = CONFIG["max_same_direction"]

    sym_data = {}
    for sym in symbols:
        df = load_data(sym).copy()
        sig_long, sig_short, stop_long, stop_short = cand_momentum(df)
        df = df.sort_index()
        sym_data[sym] = {
            "df": df,
            "sig_long_arr": sig_long.fillna(False).values,
            "sig_short_arr": sig_short.fillna(False).values,
            "stop_long_arr": stop_long.values,
            "stop_short_arr": stop_short.values,
            "opens": df["open"].values, "highs": df["high"].values,
            "lows": df["low"].values, "closes": df["close"].values,
            "times": df.index, "n": len(df),
        }

    all_events = []
    for sym, d in sym_data.items():
        for i in range(d["n"]):
            t = d["times"][i]
            if test_start <= t < test_end:
                all_events.append((t, sym, i))
    all_events.sort()
    events_by_time = defaultdict(list)
    for t, sym, i in all_events:
        events_by_time[t].append((sym, i))
    sorted_times = sorted(events_by_time.keys())

    wallet = initial_wallet
    peak_wallet = initial_wallet
    base_risk = risk_cfg.base_risk_pct
    cur_risk = base_risk
    consec_losses = 0; consec_wins = 0
    throttle_remaining = 0
    pause_until = None
    monthly_start = initial_wallet; weekly_start = initial_wallet; daily_start = initial_wallet
    if sorted_times:
        cur_month = sorted_times[0].month
        cur_week = sorted_times[0].isocalendar()[1]
        cur_day = sorted_times[0].day
    else:
        cur_month = 1; cur_week = 1; cur_day = 1
    monthly_stopped = False; weekly_stopped = False; daily_stopped = False; dd_stopped = False

    open_positions = {}
    trades = []
    ai_decisions = {"approved": 0, "rejected": 0, "skipped_rm": 0, "skipped_portfolio": 0}
    slip_frac = COSTS.slippage_bps / 10000.0
    max_bars = risk_cfg.max_bars

    for t in sorted_times:
        if t.day != cur_day:
            daily_stopped = False; daily_start = wallet; cur_day = t.day
        if t.isocalendar()[1] != cur_week:
            weekly_stopped = False; weekly_start = wallet; cur_week = t.isocalendar()[1]
        if t.month != cur_month:
            monthly_stopped = False; monthly_start = wallet; cur_month = t.month
        if wallet > peak_wallet: peak_wallet = wallet

        allowed = True; eff_risk = cur_risk
        if monthly_stopped or weekly_stopped or daily_stopped or dd_stopped: allowed = False
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

        for sym, i in events_by_time[t]:
            d = sym_data[sym]

            # === EXIT CHECK FOR OPEN POSITION ===
            if sym in open_positions:
                pos = open_positions[sym]
                if i < d["n"]:
                    h = d["highs"][i]; l = d["lows"][i]; c = d["closes"][i]
                    bars_held = i - pos["entry_bar_idx"]

                    # === BUG FIX: Update best_price IN THE POSITION DICT ===
                    if pos["side"] == "long":
                        if h > pos["best_price"]:
                            pos["best_price"] = h
                    else:
                        if l < pos["best_price"]:
                            pos["best_price"] = l

                    # === BUG FIX: Use persisted cur_trail_stop from pos ===
                    cur_trail_stop = pos.get("cur_trail_stop", pos["sp"])
                    best_price = pos["best_price"]

                    if risk_cfg.enable_trailing:
                        r_best = (best_price - pos["entry_price"]) / pos["stop_distance"] if pos["side"] == "long" \
                                 else (pos["entry_price"] - best_price) / pos["stop_distance"]
                        # Move to breakeven at +0.5R
                        if r_best >= risk_cfg.trail_trigger_1:
                            if pos["side"] == "long":
                                if pos["entry_price"] > cur_trail_stop:
                                    cur_trail_stop = pos["entry_price"]
                            else:
                                if pos["entry_price"] < cur_trail_stop:
                                    cur_trail_stop = pos["entry_price"]
                        # Trail at +1.0R
                        if r_best >= risk_cfg.trail_trigger_2:
                            if pos["side"] == "long":
                                new_stop = best_price - risk_cfg.trail_distance_r * pos["stop_distance"]
                                if new_stop > cur_trail_stop:
                                    cur_trail_stop = new_stop
                            else:
                                new_stop = best_price + risk_cfg.trail_distance_r * pos["stop_distance"]
                                if new_stop < cur_trail_stop:
                                    cur_trail_stop = new_stop

                    # === BUG FIX: Persist cur_trail_stop back to pos ===
                    pos["cur_trail_stop"] = cur_trail_stop

                    if pos["side"] == "long":
                        hit_stop = l <= cur_trail_stop; hit_tgt = h >= pos["target_price"]
                    else:
                        hit_stop = h >= cur_trail_stop; hit_tgt = l <= pos["target_price"]

                    exit_now = False; exit_price = 0; exit_reason = ""
                    if hit_stop and hit_tgt:
                        # Conservative: assume stop first
                        exit_price = cur_trail_stop; exit_reason = "stop"; exit_now = True
                    elif hit_stop:
                        exit_reason = "trail_win" if cur_trail_stop >= pos["entry_price"] else "stop"
                        exit_price = cur_trail_stop; exit_now = True
                    elif hit_tgt:
                        exit_price = pos["target_price"]; exit_reason = "target"; exit_now = True
                    elif bars_held >= max_bars:
                        exit_price = c; exit_reason = "time"; exit_now = True
                    elif t >= test_end:
                        exit_price = c; exit_reason = "end_window"; exit_now = True

                    if exit_now:
                        if pos["side"] == "long":
                            exit_fill = exit_price * (1 - slip_frac)
                            gross_pnl = pos["size"] * (exit_fill - pos["entry_price"])
                        else:
                            exit_fill = exit_price * (1 + slip_frac)
                            gross_pnl = pos["size"] * (pos["entry_price"] - exit_fill)
                        exit_notional = pos["size"] * exit_fill
                        exit_commission = exit_notional * COSTS.commission_pct
                        net_pnl = gross_pnl - exit_commission
                        r_mult_actual = net_pnl / pos["risk_amount"] if pos["risk_amount"] > 0 else 0
                        new_wallet = pos["wallet_after_entry"] + net_pnl
                        won = net_pnl > 0
                        if won:
                            consec_wins += 1; consec_losses = 0
                            cur_risk = min(cur_risk * risk_cfg.anti_mart_win_mult, risk_cfg.anti_mart_max_risk)
                        else:
                            consec_losses += 1; consec_wins = 0
                            cur_risk = max(cur_risk * risk_cfg.anti_mart_loss_mult, risk_cfg.anti_mart_min_risk)
                        if consec_losses >= risk_cfg.consec_loss_throttle_2:
                            pause_until = t + pd.Timedelta(hours=risk_cfg.consec_loss_throttle_2_hours)
                            consec_losses = 0
                        elif consec_losses >= risk_cfg.consec_loss_throttle_1:
                            throttle_remaining = risk_cfg.consec_loss_throttle_1_trades
                            cur_risk = max(cur_risk * 0.5, risk_cfg.anti_mart_min_risk)

                        trades.append({
                            "symbol": sym,
                            "entry_time": pos["entry_time"], "exit_time": t,
                            "side": pos["side"], "entry_price": pos["entry_price"],
                            "exit_price": exit_fill,
                            "stop": pos["sp"], "target": pos["target_price"],
                            "trail_stop": cur_trail_stop,  # Log the actual trail stop used
                            "best_price": best_price,      # Log best price seen
                            "size": pos["size"],
                            "risk_pct": pos["risk_pct"], "risk_usd": pos["risk_amount"],
                            "gross_pnl": gross_pnl, "cost": pos["entry_commission"] + exit_commission,
                            "net_pnl": net_pnl,
                            "wallet_before": pos["wallet_before"], "wallet_after": new_wallet,
                            "R_multiple": r_mult_actual, "exit_reason": exit_reason,
                            "bars_held": bars_held,
                            "ai_proba": pos.get("ai_proba", 0),
                        })
                        wallet = new_wallet
                        if wallet > peak_wallet: peak_wallet = wallet
                        del open_positions[sym]

            # === NEW ENTRY CHECK ===
            if not allowed:
                ai_decisions["skipped_rm"] += 1
                continue
            if sym in open_positions:
                continue
            if len(open_positions) >= max_concurrent:
                ai_decisions["skipped_portfolio"] += 1
                continue
            long_count = sum(1 for p in open_positions.values() if p["side"] == "long")
            short_count = sum(1 for p in open_positions.values() if p["side"] == "short")

            signal_long = i+1 < d["n"] and d["sig_long_arr"][i] and np.isfinite(d["opens"][i+1]) and np.isfinite(d["stop_long_arr"][i])
            signal_short = i+1 < d["n"] and d["sig_short_arr"][i] and np.isfinite(d["opens"][i+1]) and np.isfinite(d["stop_short_arr"][i])
            chosen = None
            if signal_long and signal_short: chosen = "long"
            elif signal_long: chosen = "long"
            elif signal_short: chosen = "short"
            if chosen is None: continue
            if chosen == "long" and long_count >= max_same_direction: continue
            if chosen == "short" and short_count >= max_same_direction: continue

            # AI gatekeeper
            ai = ai_per_symbol.get(sym)
            threshold = threshold_per_symbol.get(sym, 0.5)
            if ai is not None:
                feats = extract_features(d["df"], sym, i, chosen)
                X = np.array([[feats[c] for c in FEATURE_COLS]])
                proba = float(ai.predict_proba(X)[0][1])
                if proba < threshold:
                    ai_decisions["rejected"] += 1
                    continue
            else:
                proba = 1.0
            ai_decisions["approved"] += 1

            entry_raw = float(d["opens"][i+1])
            if entry_raw <= 0: continue
            if chosen == "long":
                sp_ref = float(d["stop_long_arr"][i])
                if not np.isfinite(sp_ref) or sp_ref >= entry_raw: continue
                entry_price = entry_raw * (1 + slip_frac)
                stop_distance = abs(entry_raw - sp_ref)
                sp = entry_price - stop_distance
                target_price = entry_price + 2.0 * (entry_price - sp)
            else:
                sp_ref = float(d["stop_short_arr"][i])
                if not np.isfinite(sp_ref) or sp_ref <= entry_raw: continue
                entry_price = entry_raw * (1 - slip_frac)
                stop_distance = abs(sp_ref - entry_raw)
                sp = entry_price + stop_distance
                target_price = entry_price - 2.0 * (sp - entry_price)
            if stop_distance <= 0: continue

            risk_amount = eff_risk * wallet
            size_units = risk_amount / stop_distance
            if size_units <= 0: continue

            entry_notional = size_units * entry_price
            entry_commission = entry_notional * COSTS.commission_pct
            wallet_after_entry = wallet - entry_commission

            open_positions[sym] = {
                "side": chosen, "entry_price": entry_price, "sp": sp,
                "target_price": target_price, "size": size_units,
                "entry_time": d["times"][i+1], "entry_bar_idx": i+1,
                "stop_distance": stop_distance,
                "best_price": entry_price,      # Will be updated each bar
                "cur_trail_stop": sp,           # Will be updated each bar
                "risk_amount": risk_amount, "risk_pct": eff_risk,
                "wallet_before": wallet, "wallet_after_entry": wallet_after_entry,
                "entry_commission": entry_commission, "ai_proba": proba,
            }

    # Close remaining at end of window
    for sym in list(open_positions.keys()):
        pos = open_positions[sym]
        d = sym_data[sym]
        i = d["n"] - 1
        c = d["closes"][i]
        if pos["side"] == "long":
            exit_fill = c * (1 - slip_frac)
            gross_pnl = pos["size"] * (exit_fill - pos["entry_price"])
        else:
            exit_fill = c * (1 + slip_frac)
            gross_pnl = pos["size"] * (pos["entry_price"] - exit_fill)
        exit_notional = pos["size"] * exit_fill
        exit_commission = exit_notional * COSTS.commission_pct
        net_pnl = gross_pnl - exit_commission
        r_mult_actual = net_pnl / pos["risk_amount"] if pos["risk_amount"] > 0 else 0
        new_wallet = pos["wallet_after_entry"] + net_pnl
        trades.append({
            "symbol": sym,
            "entry_time": pos["entry_time"], "exit_time": d["times"][i],
            "side": pos["side"], "entry_price": pos["entry_price"], "exit_price": exit_fill,
            "stop": pos["sp"], "target": pos["target_price"],
            "trail_stop": pos.get("cur_trail_stop", pos["sp"]),
            "best_price": pos.get("best_price", pos["entry_price"]),
            "size": pos["size"],
            "risk_pct": pos["risk_pct"], "risk_usd": pos["risk_amount"],
            "gross_pnl": gross_pnl, "cost": pos["entry_commission"] + exit_commission,
            "net_pnl": net_pnl,
            "wallet_before": pos["wallet_before"], "wallet_after": new_wallet,
            "R_multiple": r_mult_actual, "exit_reason": "end_window",
            "bars_held": i - pos["entry_bar_idx"], "ai_proba": pos.get("ai_proba", 0),
        })
        wallet = new_wallet
        del open_positions[sym]

    if trades:
        wallets = [initial_wallet] + [t["wallet_after"] for t in trades]
        peak_arr = np.maximum.accumulate(wallets)
        dd_arr = (peak_arr - wallets) / np.where(peak_arr > 0, peak_arr, 1)
        max_dd = float(np.max(dd_arr))
    else:
        max_dd = 0
    n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
    n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
    wr = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0

    # Count exit reasons
    exit_reasons = {}
    for t in trades:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return trades, {
        "n_trades": len(trades), "final_wallet": wallet,
        "net": wallet - initial_wallet, "net_pct": (wallet - initial_wallet) / initial_wallet,
        "peak_wallet": peak_wallet, "max_dd_pct": max_dd,
        "n_wins": n_wins, "n_losses": n_losses, "win_rate": wr,
        "ai_decisions": ai_decisions,
        "exit_reasons": exit_reasons,
    }


# ============================================================================
# Coin flip AI class
# ============================================================================

class CoinFlipAI:
    def predict_proba(self, X):
        n = X.shape[0]
        proba = np.random.rand(n, 2)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


# ============================================================================
# Run 3 versions × 6 months
# ============================================================================

def main():
    print("="*120)
    print("HONEST RE-TEST — Fixed trailing stop bug, 3 versions side by side")
    print(f"6 months: {TEST_MONTHS[0]['label']} to {TEST_MONTHS[-1]['label']}")
    print("="*120)

    print("\nBUG FIXED:")
    print("  pos['best_price'] now updated every bar (was: never updated, always = entry_price)")
    print("  pos['cur_trail_stop'] now persists across bars (was: local variable, reset each bar)")
    print("  This means trailing stops NOW ACTUALLY WORK:")
    print("    At +0.5R favorable: stop moves to breakeven")
    print("    At +1.0R favorable: stop trails at 0.3R below best price")
    print("  Previously: stop was ALWAYS the original 0.5-ATR stop, never moved")

    risk_cfg = make_risk_cfg()

    results = {"with_ai": [], "no_ai": [], "coinflip": []}

    for mi, month in enumerate(TEST_MONTHS):
        print(f"\n{'='*100}")
        print(f"Month {mi+1}/6: {month['label']}")
        print(f"  Train: {month['train_start'].date()} -> {month['train_end'].date()}")
        print(f"  Test:  {month['test_start'].date()} -> {month['test_end'].date()}")
        print(f"{'='*100}")

        # Train AIs for this month
        print(f"\n  Training AIs...", flush=True)
        ai_per_symbol = {}
        threshold_per_symbol = {}
        for sym in CONFIG["symbols"]:
            try:
                samples = generate_training_samples(sym, month['train_start'], month['train_end'])
                if len(samples) < 30: continue
                X = np.array([[s[c] for c in FEATURE_COLS] for s in samples])
                y = np.array([s["label"] for s in samples])
                model = RandomForestClassifier(
                    n_estimators=80, max_depth=5, min_samples_leaf=15,
                    random_state=42, class_weight="balanced")
                model.fit(X, y)
                ai_per_symbol[sym] = model
                # Validate threshold
                val_samples = generate_training_samples(sym, month['val_start'], month['val_end'])
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
            except Exception as e:
                print(f"    {sym}: FAILED - {e}", flush=True)

        # Version A: WITH AI
        print(f"\n  A) WITH AI (real Random Forest gatekeeper):", flush=True)
        trades_a, stats_a = backtest_month_FIXED(
            ai_per_symbol, threshold_per_symbol, risk_cfg,
            month['test_start'], month['test_end'])
        print(f"     {stats_a['n_trades']} trades, WR={stats_a['win_rate']*100:.1f}%, "
              f"net=${stats_a['net']:+,.2f} ({stats_a['net_pct']*100:+.2f}%), DD={stats_a['max_dd_pct']*100:.1f}%", flush=True)
        print(f"     Exit reasons: {stats_a['exit_reasons']}", flush=True)
        results["with_ai"].append({"month": month['label'], **stats_a})

        _IND_CACHE.clear(); gc.collect()

        # Version B: NO AI (take every signal)
        print(f"\n  B) NO AI (take every momentum bar candidate):", flush=True)
        trades_b, stats_b = backtest_month_FIXED(
            {}, {}, risk_cfg,
            month['test_start'], month['test_end'])
        print(f"     {stats_b['n_trades']} trades, WR={stats_b['win_rate']*100:.1f}%, "
              f"net=${stats_b['net']:+,.2f} ({stats_b['net_pct']*100:+.2f}%), DD={stats_b['max_dd_pct']*100:.1f}%", flush=True)
        print(f"     Exit reasons: {stats_b['exit_reasons']}", flush=True)
        results["no_ai"].append({"month": month['label'], **stats_b})

        _IND_CACHE.clear(); gc.collect()

        # Version C: COIN FLIP
        print(f"\n  C) COIN FLIP (random 50/50 AI):", flush=True)
        coinflip_ai = {sym: CoinFlipAI() for sym in CONFIG["symbols"]}
        coinflip_thresh = {sym: 0.5 for sym in CONFIG["symbols"]}
        trades_c, stats_c = backtest_month_FIXED(
            coinflip_ai, coinflip_thresh, risk_cfg,
            month['test_start'], month['test_end'])
        print(f"     {stats_c['n_trades']} trades, WR={stats_c['win_rate']*100:.1f}%, "
              f"net=${stats_c['net']:+,.2f} ({stats_c['net_pct']*100:+.2f}%), DD={stats_c['max_dd_pct']*100:.1f}%", flush=True)
        print(f"     Exit reasons: {stats_c['exit_reasons']}", flush=True)
        results["coinflip"].append({"month": month['label'], **stats_c})

        del ai_per_symbol, coinflip_ai
        _IND_CACHE.clear(); gc.collect()

    # Summary
    print("\n" + "="*120)
    print("6-MONTH SUMMARY — FIXED TRAILING STOP, 3 VERSIONS")
    print("="*120)
    print(f"\n{'Month':<10} {'WITH AI':>25} {'NO AI':>25} {'COIN FLIP':>25}")
    print(f"{'':<10} {'trades/WR/net%/DD%':>25} {'trades/WR/net%/DD%':>25} {'trades/WR/net%/DD%':>25}")
    print("-"*90)
    for i in range(len(TEST_MONTHS)):
        a = results["with_ai"][i]; b = results["no_ai"][i]; c = results["coinflip"][i]
        print(f"{a['month']:<10} "
              f"{a['n_trades']:>3}t/{a['win_rate']*100:>4.0f}%/{a['net_pct']*100:>+6.1f}%/{a['max_dd_pct']*100:>4.1f}%   "
              f"{b['n_trades']:>3}t/{b['win_rate']*100:>4.0f}%/{b['net_pct']*100:>+6.1f}%/{b['max_dd_pct']*100:>4.1f}%   "
              f"{c['n_trades']:>3}t/{c['win_rate']*100:>4.0f}%/{c['net_pct']*100:>+6.1f}%/{c['max_dd_pct']*100:>4.1f}%")

    # Averages
    print(f"\n{'AVERAGE':<10}", end="")
    for v in ["with_ai", "no_ai", "coinflip"]:
        avg_trades = np.mean([r['n_trades'] for r in results[v]])
        avg_wr = np.mean([r['win_rate'] for r in results[v] if r['n_trades'] > 0])
        avg_net = np.mean([r['net_pct'] for r in results[v]])
        avg_dd = np.mean([r['max_dd_pct'] for r in results[v]])
        n_prof = sum(1 for r in results[v] if r['net'] > 0)
        print(f"  {avg_trades:>3.0f}t/{avg_wr*100:>4.0f}%/{avg_net*100:>+6.1f}%/{avg_dd*100:>4.1f}% ({n_prof}/6)", end="")
    print()

    # Exit reason breakdown
    print(f"\n{'EXIT REASONS (avg across 6 months):'}")
    for v_name in ["with_ai", "no_ai", "coinflip"]:
        all_reasons = {}
        for r in results[v_name]:
            for reason, count in r.get("exit_reasons", {}).items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
        total = sum(all_reasons.values())
        print(f"  {v_name:>10}: ", end="")
        for reason in sorted(all_reasons.keys()):
            print(f"{reason}={all_reasons[reason]} ({all_reasons[reason]/total*100:.0f}%)  ", end="")
        print()

    # Verdict
    print("\n" + "="*100)
    print("HONEST VERDICT")
    print("="*100)
    avg_ai = np.mean([r['net_pct'] for r in results["with_ai"]])
    avg_noai = np.mean([r['net_pct'] for r in results["no_ai"]])
    avg_cf = np.mean([r['net_pct'] for r in results["coinflip"]])

    print(f"\n  WITH AI avg monthly:    {avg_ai*100:+.2f}%")
    print(f"  NO AI avg monthly:      {avg_noai*100:+.2f}%")
    print(f"  COIN FLIP avg monthly:  {avg_cf*100:+.2f}%")

    if abs(avg_ai - avg_noai) < 0.05:
        print("\n  → AI adds NO value. The edge is 100% from momentum bars + risk management.")
    elif avg_ai > avg_noai * 1.5:
        print("\n  → AI adds SIGNIFICANT value (>50% improvement over no-AI).")
    else:
        print(f"\n  → AI adds MARGINAL value ({(avg_ai/avg_noai-1)*100:.0f}% improvement over no-AI).")

    if avg_cf > 0:
        print("  → COIN FLIP is profitable — edge doesn't come from AI predictions.")
    else:
        print("  → COIN FLIP loses money — AI predictions DO matter.")

    # Save results
    out_path = DOWNLOAD / "honest_retest_6month.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n>>> Wrote {out_path}")


if __name__ == "__main__":
    main()
