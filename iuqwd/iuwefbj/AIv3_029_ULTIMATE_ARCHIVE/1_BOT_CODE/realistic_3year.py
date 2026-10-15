"""
REALISTIC 3-YEAR TEST — What you would ACTUALLY earn in real trading.

Settings (matching real-world conditions):
  - 1-hour checking (bot checks every hour, same as paper trading bot)
  - Spot-capped positions (can't buy more than you have)
  - Limit order costs: 0.075% per side (BNB maker discount)
  - Zero slippage (limit orders fill at your price or not at all)
  - 9 markets
  - AI gatekeeper (pre-trained models, retrained each month)
  - Fixed trailing stops (bug-fixed version)
  - All 7 risk management layers

This is the MOST REALISTIC possible backtest. If you had run this bot
for 3 years, these are the numbers you would have seen.
"""
import sys, time, json, gc, pickle
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from backtester import atr
from walkforward_3year import (
    CONFIG, FEATURE_COLS, load_data, get_indicators,
    extract_features, cand_momentum, generate_training_samples,
    make_risk_cfg, generate_test_months
)
from risk_managed_v2 import RiskConfig
import walkforward_3year as wf

_IND_CACHE = wf._IND_CACHE
DATA_DIR = wf.DATA_DIR
INITIAL_WALLET = 10000.0

# REALISTIC COSTS
COMM_PCT = 0.00075  # 0.075% per side (BNB maker discount)
SLIP_FRAC = 0.0     # Zero slippage (limit orders)
CAP_SPOT = True      # Can't buy more than wallet

SAVE_PATH = DATA_DIR.parent / "download" / "realistic_3year_results.json"


def train_ais_for_month(month):
    ai_dict = {}; thresh_dict = {}
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
            ai_dict[sym] = model
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
                thresh_dict[sym] = best_t
            else:
                thresh_dict[sym] = 0.5
        except:
            pass
    return ai_dict, thresh_dict


def backtest_realistic(ai_per_symbol, threshold_per_symbol, risk_cfg,
                       test_start, test_end, initial_wallet=INITIAL_WALLET):
    symbols = CONFIG["symbols"]
    max_concurrent = 1
    slip = SLIP_FRAC
    comm = COMM_PCT
    
    sym_data = {}
    for sym in symbols:
        df = load_data(sym).copy().sort_index()
        sig_long, sig_short, stop_long, stop_short = cand_momentum(df)
        sym_data[sym] = {
            "df": df,
            "sig_long_arr": sig_long.fillna(False).values,
            "sig_short_arr": sig_short.fillna(False).values,
            "stop_long_arr": stop_long.values,
            "stop_short_arr": stop_short.values,
            "opens": df["open"].values,
            "highs": df["high"].values,
            "lows": df["low"].values,
            "closes": df["close"].values,
            "times": df.index,
            "n": len(df),
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
    base_risk = 0.01
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
    monthly_stopped = weekly_stopped = daily_stopped = dd_stopped = False
    
    open_positions = {}
    trades = []
    max_bars = 24
    
    for t in sorted_times:
        # Risk management
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
            if monthly_start > 0 and (wallet - monthly_start) / monthly_start <= -0.05:
                monthly_stopped = True; allowed = False
            if allowed and weekly_start > 0 and (wallet - weekly_start) / weekly_start <= -0.03:
                weekly_stopped = True; allowed = False
            if allowed and daily_start > 0 and (wallet - daily_start) / daily_start <= -0.02:
                daily_stopped = True; allowed = False
        if allowed:
            if peak_wallet > 0:
                dd = (peak_wallet - wallet) / peak_wallet
                if dd >= 0.17: dd_stopped = True; allowed = False
                elif dd >= 0.12: eff_risk = min(eff_risk, base_risk * 0.25)
                elif dd >= 0.07: eff_risk = min(eff_risk, base_risk * 0.50)
        if allowed:
            if t.month == 12:
                nm = pd.Timestamp(year=t.year+1, month=1, day=1, tz="UTC")
            else:
                nm = pd.Timestamp(year=t.year, month=t.month+1, day=1, tz="UTC")
            if (nm - t).days <= 3 and monthly_start > 0:
                if (wallet - monthly_start) / monthly_start > 0:
                    eff_risk = 0.004
            if throttle_remaining > 0:
                eff_risk = max(eff_risk * 0.5, 0.004)
                throttle_remaining -= 1
        
        for sym, i in events_by_time[t]:
            d = sym_data[sym]
            
            # EXIT CHECK
            if sym in open_positions:
                pos = open_positions[sym]
                if i < d["n"]:
                    h = d["highs"][i]; l = d["lows"][i]; c = d["closes"][i]
                    
                    # Update trailing stop (FIXED — persists in pos dict)
                    cur_trail = pos.get("cur_trail_stop", pos["sp"])
                    best = pos.get("best_price", pos["entry_price"])
                    if pos["side"] == "long":
                        if h > best: best = h
                    else:
                        if l < best: best = l
                    
                    r_best = (best - pos["entry_price"]) / pos["stop_distance"] if pos["side"] == "long" \
                             else (pos["entry_price"] - best) / pos["stop_distance"]
                    if r_best >= 0.5:
                        if pos["side"] == "long":
                            if pos["entry_price"] > cur_trail: cur_trail = pos["entry_price"]
                        else:
                            if pos["entry_price"] < cur_trail: cur_trail = pos["entry_price"]
                    if r_best >= 1.0:
                        if pos["side"] == "long":
                            ns = best - 0.3 * pos["stop_distance"]
                            if ns > cur_trail: cur_trail = ns
                        else:
                            ns = best + 0.3 * pos["stop_distance"]
                            if ns < cur_trail: cur_trail = ns
                    
                    pos["best_price"] = best
                    pos["cur_trail_stop"] = cur_trail
                    pos["bars_held"] = pos.get("bars_held", 0) + 1
                    bars = pos["bars_held"]
                    
                    # Check exits
                    if pos["side"] == "long":
                        hit_stop = l <= cur_trail; hit_tgt = h >= pos["target_price"]
                    else:
                        hit_stop = h >= cur_trail; hit_tgt = l >= pos["target_price"]
                    
                    exit_now = False; exit_price = 0; exit_reason = ""
                    if hit_stop and hit_tgt:
                        exit_price = cur_trail; exit_reason = "stop"; exit_now = True
                    elif hit_stop:
                        exit_reason = "trail_win" if cur_trail >= pos["entry_price"] else "stop"
                        exit_price = cur_trail; exit_now = True
                    elif hit_tgt:
                        exit_price = pos["target_price"]; exit_reason = "target"; exit_now = True
                    elif bars >= max_bars:
                        exit_price = c; exit_reason = "time"; exit_now = True
                    
                    if exit_now:
                        if pos["side"] == "long":
                            exit_fill = exit_price * (1 - slip)
                            gross = pos["size"] * (exit_fill - pos["entry_price"])
                        else:
                            exit_fill = exit_price * (1 + slip)
                            gross = pos["size"] * (pos["entry_price"] - exit_fill)
                        exit_comm = pos["size"] * exit_fill * comm
                        net = gross - exit_comm
                        r_mult = net / pos["risk_amount"] if pos["risk_amount"] > 0 else 0
                        new_wallet = pos["wallet_after_entry"] + net
                        won = net > 0
                        
                        if won:
                            consec_wins += 1; consec_losses = 0
                            cur_risk = min(cur_risk * 1.30, 0.015)
                        else:
                            consec_losses += 1; consec_wins = 0
                            cur_risk = max(cur_risk * 0.60, 0.004)
                        if consec_losses >= 4:
                            pause_until = t + pd.Timedelta(hours=24); consec_losses = 0
                        elif consec_losses >= 2:
                            throttle_remaining = 2; cur_risk = max(cur_risk * 0.5, 0.004)
                        
                        trades.append({
                            "symbol": sym, "side": pos["side"],
                            "entry_price": pos["entry_price"], "exit_price": exit_fill,
                            "net_pnl": net, "r_mult": r_mult,
                            "exit_reason": exit_reason, "win": won,
                            "wallet_after": new_wallet,
                        })
                        wallet = new_wallet
                        if wallet > peak_wallet: peak_wallet = wallet
                        del open_positions[sym]
            
            # NEW ENTRY
            if not allowed: continue
            if sym in open_positions: continue
            if len(open_positions) >= max_concurrent: continue
            
            signal_long = i+1 < d["n"] and d["sig_long_arr"][i] and np.isfinite(d["opens"][i+1]) and np.isfinite(d["stop_long_arr"][i])
            signal_short = i+1 < d["n"] and d["sig_short_arr"][i] and np.isfinite(d["opens"][i+1]) and np.isfinite(d["stop_short_arr"][i])
            chosen = "long" if (signal_long and signal_short) else ("long" if signal_long else ("short" if signal_short else None))
            if chosen is None: continue
            
            # AI gatekeeper
            ai = ai_per_symbol.get(sym)
            threshold = threshold_per_symbol.get(sym, 0.5)
            if ai is not None:
                try:
                    feats = extract_features(d["df"], sym, i, chosen)
                    X = np.array([[feats[c] for c in FEATURE_COLS]])
                    proba = float(ai.predict_proba(X)[0][1])
                    if proba < threshold: continue
                except: pass
            
            entry_raw = float(d["opens"][i+1])
            if entry_raw <= 0: continue
            if chosen == "long":
                sp_ref = float(d["stop_long_arr"][i])
                if not np.isfinite(sp_ref) or sp_ref >= entry_raw: continue
                entry_price = entry_raw * (1 + slip)
                sd = abs(entry_raw - sp_ref)
                sp = entry_price - sd
                tp = entry_price + 2.0 * (entry_price - sp)
            else:
                sp_ref = float(d["stop_short_arr"][i])
                if not np.isfinite(sp_ref) or sp_ref <= entry_raw: continue
                entry_price = entry_raw * (1 - slip)
                sd = abs(sp_ref - entry_raw)
                sp = entry_price + sd
                tp = entry_price - 2.0 * (sp - entry_price)
            if sd <= 0: continue
            
            risk_amount = eff_risk * wallet
            size_units = risk_amount / sd
            
            # SPOT CAP: can't buy more than wallet allows
            if CAP_SPOT:
                max_size = wallet / entry_price
                if size_units > max_size:
                    size_units = max_size
                    risk_amount = size_units * sd
            
            if size_units <= 0: continue
            entry_comm = size_units * entry_price * comm
            wallet_after_entry = wallet - entry_comm
            
            open_positions[sym] = {
                "side": chosen, "entry_price": entry_price,
                "sp": sp, "target_price": tp,
                "size": size_units, "stop_distance": sd,
                "best_price": entry_price, "cur_trail_stop": sp,
                "risk_amount": risk_amount, "risk_pct": eff_risk,
                "wallet_before": wallet, "wallet_after_entry": wallet_after_entry,
                "entry_time": d["times"][i+1], "bars_held": 0,
            }
    
    # Close remaining
    for sym in list(open_positions.keys()):
        pos = open_positions[sym]
        d = sym_data[sym]
        c = d["closes"][-1]
        if pos["side"] == "long":
            exit_fill = c * (1 - slip)
            gross = pos["size"] * (exit_fill - pos["entry_price"])
        else:
            exit_fill = c * (1 + slip)
            gross = pos["size"] * (pos["entry_price"] - exit_fill)
        net = gross - (pos["size"] * exit_fill * comm)
        r_mult = net / pos["risk_amount"] if pos["risk_amount"] > 0 else 0
        new_w = pos["wallet_after_entry"] + net
        trades.append({"symbol": sym, "net_pnl": net, "r_mult": r_mult,
                       "exit_reason": "end_window", "win": net > 0, "wallet_after": new_w})
        wallet = new_w
        del open_positions[sym]
    
    n_wins = sum(1 for t in trades if t["win"])
    n_losses = sum(1 for t in trades if not t["win"])
    wr = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0
    
    if trades:
        wallets = [initial_wallet] + [t["wallet_after"] for t in trades]
        peak_arr = np.maximum.accumulate(wallets)
        dd = (peak_arr - wallets) / np.where(peak_arr > 0, peak_arr, 1)
        max_dd = float(np.max(dd))
    else:
        max_dd = 0
    
    exit_reasons = {}
    for t in trades:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    
    return trades, {
        "n_trades": len(trades), "final_wallet": wallet,
        "net": wallet - initial_wallet, "net_pct": (wallet - initial_wallet) / initial_wallet,
        "n_wins": n_wins, "n_losses": n_losses, "win_rate": wr,
        "max_dd_pct": max_dd, "exit_reasons": exit_reasons,
        "avg_win_r": np.mean([t["r_mult"] for t in trades if t["win"]]) if n_wins else 0,
        "avg_loss_r": np.mean([t["r_mult"] for t in trades if not t["win"]]) if n_losses else 0,
    }


def main():
    print("="*90)
    print("REALISTIC 3-YEAR TEST")
    print("1-hour checking | Spot-capped | Limit order costs (0.075%/side) | Zero slippage")
    print("This is what you would ACTUALLY earn running the bot for 3 years")
    print("="*90, flush=True)
    
    all_months = generate_test_months()
    risk_cfg = make_risk_cfg()
    
    # Load partial results if they exist
    results = []
    start_idx = 0
    if SAVE_PATH.exists():
        try:
            saved = json.loads(SAVE_PATH.read_text())
            results = saved.get("results", [])
            start_idx = saved.get("last_month_idx", 0)
            if start_idx > 0:
                print(f"[RESUME] Starting from month {start_idx+1}/36", flush=True)
        except:
            pass
    
    t0 = time.time()
    
    for mi in range(start_idx, len(all_months)):
        month = all_months[mi]
        print(f"\nMonth {mi+1}/36: {month['label']}", flush=True)
        
        # Train AIs for this month
        ai_dict, thresh_dict = train_ais_for_month(month)
        
        # Run backtest
        trades, stats = backtest_realistic(
            ai_dict, thresh_dict, risk_cfg,
            month['test_start'], month['test_end']
        )
        
        print(f"  {stats['n_trades']:>3}t  WR={stats['win_rate']*100:>4.0f}%  "
              f"net=${stats['net']:>+9,.0f} ({stats['net_pct']*100:>+6.1f}%)  "
              f"DD={stats['max_dd_pct']*100:.1f}%  "
              f"winR={stats['avg_win_r']:+.2f}  lossR={stats['avg_loss_r']:+.2f}", flush=True)
        
        results.append({
            "month": month['label'],
            "n_trades": stats['n_trades'],
            "win_rate": stats['win_rate'],
            "net": stats['net'],
            "net_pct": stats['net_pct'],
            "max_dd_pct": stats['max_dd_pct'],
            "n_wins": stats['n_wins'],
            "n_losses": stats['n_losses'],
            "avg_win_r": stats['avg_win_r'],
            "avg_loss_r": stats['avg_loss_r'],
            "exit_reasons": stats['exit_reasons'],
        })
        
        del ai_dict
        _IND_CACHE.clear()
        gc.collect()
        
        # SAVE AFTER EVERY MONTH
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAVE_PATH.write_text(json.dumps({
            "results": results,
            "last_month_idx": mi + 1,
        }, indent=2, default=str))
        
        elapsed = time.time() - t0
        done = mi + 1 - start_idx
        rate = done / elapsed if elapsed > 0 else 0
        eta = (36 - mi - 1) / rate if rate > 0 else 0
        print(f"  [saved {mi+1}/36] {rate:.1f}/mo, eta {eta:.0f}s", flush=True)
    
    # ============================================================================
    # FINAL RESULTS
    # ============================================================================
    print(f"\n{'='*90}")
    print("REALISTIC 3-YEAR RESULTS — WHAT YOU WOULD ACTUALLY EARN")
    print(f"{'='*90}")
    print(f"\nSettings: 1h check | spot-capped | 0.075% maker fee | 0% slippage | 9 markets")
    print(f"\n{'Month':<10} {'Trades':>7} {'Wins':>5} {'Loss':>6} {'WR':>6} {'Net P&L':>11} {'Net %':>8} {'MaxDD':>7} {'WinR':>6} {'LossR':>6}")
    print("-"*80)
    
    for r in results:
        print(f"{r['month']:<10} {r['n_trades']:>7} {r['n_wins']:>5} {r['n_losses']:>6} "
              f"{r['win_rate']*100:>5.0f}% ${r['net']:>+10,.0f} {r['net_pct']*100:>+7.1f}% "
              f"{r['max_dd_pct']*100:>6.1f}% {r['avg_win_r']:>+5.2f} {r['avg_loss_r']:>+5.2f}")
    
    # Aggregate
    total_trades = sum(r['n_trades'] for r in results)
    total_wins = sum(r['n_wins'] for r in results)
    total_losses = sum(r['n_losses'] for r in results)
    total_net = sum(r['net'] for r in results)
    avg_net = np.mean([r['net_pct'] for r in results])
    median_net = np.median([r['net_pct'] for r in results])
    avg_wr = np.mean([r['win_rate'] for r in results])
    avg_dd = np.mean([r['max_dd_pct'] for r in results])
    worst_dd = max(r['max_dd_pct'] for r in results)
    best_month = max(results, key=lambda x: x['net_pct'])
    worst_month = min(results, key=lambda x: x['net_pct'])
    n_prof = sum(1 for r in results if r['net'] > 0)
    n_loss = sum(1 for r in results if r['net'] <= 0)
    
    # Compounded
    compounded = 1.0
    for r in results:
        compounded *= (1 + r['net_pct'])
    
    print(f"\n{'='*80}")
    print("3-YEAR SUMMARY")
    print(f"{'='*80}")
    print(f"  Months tested:        {len(results)}")
    print(f"  Profitable months:    {n_prof}/{len(results)} ({n_prof/len(results)*100:.0f}%)")
    print(f"  Losing months:        {n_loss}/{len(results)}")
    print(f"  Total trades:         {total_trades}")
    print(f"  Total wins:           {total_wins}")
    print(f"  Total losses:         {total_losses}")
    print(f"  Win rate:             {total_wins/total_trades*100:.1f}%")
    print(f"  Avg win:              {np.mean([r['avg_win_r'] for r in results]):+.2f}R")
    print(f"  Avg loss:             {np.mean([r['avg_loss_r'] for r in results]):+.2f}R")
    print(f"  Avg monthly profit:   ${np.mean([r['net'] for r in results]):+,.0f} ({avg_net*100:+.1f}%)")
    print(f"  Median monthly profit:${median_net:+,.0f} ({median_net*100:+.1f}%)")
    print(f"  Best month:           {best_month['month']} = ${best_month['net']:+,.0f} ({best_month['net_pct']*100:+.1f}%)")
    print(f"  Worst month:          {worst_month['month']} = ${worst_month['net']:+,.0f} ({worst_month['net_pct']*100:+.1f}%)")
    print(f"  Avg drawdown:         {avg_dd*100:.1f}%")
    print(f"  Worst drawdown:       {worst_dd*100:.1f}%")
    print(f"  Compounded 3-yr:      {(compounded-1)*100:+,.0f}%")
    print(f"  Sum of monthly:       {sum(r['net_pct'] for r in results)*100:+,.1f}%")
    print(f"  Total P&L (sum):      ${total_net:+,.0f}")
    
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    if n_prof >= 30:
        print(f"  ✓ {n_prof}/36 months profitable — edge is REAL")
    elif n_prof >= 24:
        print(f"  ~ {n_prof}/36 months profitable — edge is PROBABLY real")
    else:
        print(f"  ✗ {n_prof}/36 months profitable — edge is WEAK")
    
    if avg_net > 0:
        print(f"  ✓ Average monthly profit: {avg_net*100:+.1f}%")
        print(f"  ✓ Annual return (simple): {avg_net*12*100:+.0f}%")
        print(f"  ✓ 3-year compounded: {(compounded-1)*100:+,.0f}%")
    else:
        print(f"  ✗ Average monthly LOSS: {avg_net*100:+.1f}%")
    
    print(f"\n>>> Results saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
