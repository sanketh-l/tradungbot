"""
Phase 3b + 4: Coarse-to-fine funnel and metrics.

Pipeline:
  Pass 1 (fast filter): all 2500 configs on a 6-month IS slice.
      Vectorized, ~100ms per config. Keep configs with >= 30 trades AND
      positive gross R.
  Pass 2 (full IS): survivors run on the full 70% IS window with the
      event-driven backtester. Keep configs with >= 200 closed trades.
  Pass 3 (OOS): survivors run on the untouched 30% OOS window.
      No further filtering at this stage — just measure.
  Metrics: compute all Phase 4 metrics for IS and OOS, apply gate checks,
      write results.csv.

IS/OOS split:
  IS  = earliest 70% of the full dataset (chronological)
  OOS = latest  30%
  We NEVER touch OOS during Pass 1 or Pass 2.

For Pass 1 we use the FIRST 6 months of the IS slice as a quick filter —
this is a subset of IS, so it does not contaminate OOS.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtester import Costs, BacktestResult, Trade, backtest, pass1_fast_filter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "download" / "results.csv"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

PASS1_OUT = DATA_DIR / "pass1_survivors.json"
PASS2_OUT = DATA_DIR / "pass2_results.json"
PASS3_OUT = DATA_DIR / "pass3_oos_results.json"


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_{timeframe}.parquet")
    df = df.set_index("open_time").sort_index()
    return df


def split_is_oos(df: pd.DataFrame, is_frac: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    is_end = int(n * is_frac)
    return df.iloc[:is_end], df.iloc[is_end:]


# ============================================================================
# Metrics computation
# ============================================================================

def compute_metrics(result: BacktestResult, costs_label: str = "default") -> dict:
    """Compute all Phase 4 metrics from a backtest result."""
    trades = result.trades
    n_closed = len(trades)
    if n_closed == 0:
        return {
            "n_closed": 0, "n_wins": 0, "n_losses": 0, "n_other": 0,
            "win_rate": 0.0, "avg_win_R": 0.0, "avg_loss_R": 0.0,
            "profit_factor": 0.0, "net_profit_pct": 0.0,
            "expectancy_R": 0.0, "max_dd_pct": 0.0,
            "longest_loss_streak": 0, "sharpe_daily": 0.0,
            "final_equity": result.final_equity, "costs_label": costs_label,
        }

    wins   = [t for t in trades if t.exit_reason == "target" and t.net_pnl > 0]
    losses = [t for t in trades if t.exit_reason == "stop"   and t.net_pnl < 0]
    others = [t for t in trades if t not in wins and t not in losses]

    n_wins = len(wins)
    n_losses = len(losses)
    n_other = n_closed - n_wins - n_losses
    n_wl = n_wins + n_losses
    win_rate = n_wins / n_wl if n_wl > 0 else 0.0

    win_Rs  = [t.r_multiple for t in wins]
    loss_Rs = [t.r_multiple for t in losses]
    avg_win_R  = float(np.mean(win_Rs))  if win_Rs  else 0.0
    avg_loss_R = float(np.mean(loss_Rs)) if loss_Rs else 0.0

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss   = -sum(t.net_pnl for t in losses)  # positive number
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)

    net_profit = result.final_equity - result.initial_equity
    net_profit_pct = net_profit / result.initial_equity

    expectancy_R = float(np.mean([t.r_multiple for t in trades]))

    # Max drawdown on equity curve
    eq = result.equity_curve
    if len(eq) > 0:
        peak = np.maximum.accumulate(eq)
        # Avoid div by zero
        dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
        max_dd_pct = float(np.max(dd)) if len(dd) > 0 else 0.0
    else:
        max_dd_pct = 0.0

    # Longest losing streak (consecutive trades with net_pnl < 0 OR exit_reason == "stop")
    longest_loss = 0
    cur = 0
    for t in trades:
        if t.net_pnl < 0:
            cur += 1
            longest_loss = max(longest_loss, cur)
        else:
            cur = 0

    # Daily Sharpe: aggregate trades by day, compute daily returns, Sharpe = mean/std * sqrt(365)
    if trades:
        df_t = pd.DataFrame([
            {"exit_time": t.exit_time, "pnl": t.net_pnl}
            for t in trades
        ])
        df_t["date"] = pd.to_datetime(df_t["exit_time"]).dt.date
        daily = df_t.groupby("date")["pnl"].sum()
        # Daily returns = pnl / equity_at_start_of_day. Approximate by absolute pnl
        # divided by mean equity (~1). For Sharpe ranking this is fine.
        if daily.std() > 0:
            sharpe_daily = float(daily.mean() / daily.std() * np.sqrt(365))
        else:
            sharpe_daily = 0.0
    else:
        sharpe_daily = 0.0

    return {
        "n_closed": n_closed, "n_wins": n_wins, "n_losses": n_losses,
        "n_other": n_other, "win_rate": win_rate,
        "avg_win_R": avg_win_R, "avg_loss_R": avg_loss_R,
        "profit_factor": profit_factor,
        "net_profit_pct": net_profit_pct,
        "expectancy_R": expectancy_R, "max_dd_pct": max_dd_pct,
        "longest_loss_streak": longest_loss,
        "sharpe_daily": sharpe_daily,
        "final_equity": result.final_equity,
        "costs_label": costs_label,
    }


GATE_THRESHOLDS = {
    "min_trades":       200,
    "min_win_rate":     0.70,
    "min_avg_win_R":    2.0,
    "target_avg_loss_R": -1.0,  # informational; we report deviation
    "min_profit_factor": 1.0,
    "min_net_profit":    0.0,
    "min_expectancy":    0.0,
    "max_dd":            0.25,
    "max_oos_win_drop_pp": 15.0,
}


def gate_check(metrics: dict, label: str = "OOS") -> dict[str, Any]:
    """Return a dict of gate_name -> (passed, value, threshold)."""
    g = {}
    g["gate_trades"]    = (metrics["n_closed"] >= GATE_THRESHOLDS["min_trades"],
                           metrics["n_closed"], GATE_THRESHOLDS["min_trades"])
    g["gate_win_rate"]  = (metrics["win_rate"] >= GATE_THRESHOLDS["min_win_rate"],
                           metrics["win_rate"], GATE_THRESHOLDS["min_win_rate"])
    g["gate_avg_win_R"] = (metrics["avg_win_R"] >= GATE_THRESHOLDS["min_avg_win_R"],
                           metrics["avg_win_R"], GATE_THRESHOLDS["min_avg_win_R"])
    g["gate_pf"]        = (metrics["profit_factor"] > GATE_THRESHOLDS["min_profit_factor"],
                           metrics["profit_factor"], GATE_THRESHOLDS["min_profit_factor"])
    g["gate_net_profit"] = (metrics["net_profit_pct"] > GATE_THRESHOLDS["min_net_profit"],
                            metrics["net_profit_pct"], GATE_THRESHOLDS["min_net_profit"])
    g["gate_expectancy"] = (metrics["expectancy_R"] > GATE_THRESHOLDS["min_expectancy"],
                            metrics["expectancy_R"], GATE_THRESHOLDS["min_expectancy"])
    g["gate_max_dd"]    = (metrics["max_dd_pct"] <= GATE_THRESHOLDS["max_dd"],
                           metrics["max_dd_pct"], GATE_THRESHOLDS["max_dd"])
    return g


def all_gates_passed(oos_metrics: dict) -> bool:
    g = gate_check(oos_metrics, "OOS")
    return all(passed for passed, _, _ in g.values())


# ============================================================================
# Pipeline orchestration
# ============================================================================

def run_pass1(configs: list[dict], log_every: int = 100) -> list[dict]:
    """Pass 1: fast filter on 6-month IS slice for each (symbol, timeframe)."""
    print("\n=== Pass 1: fast vectorized filter (6-month IS slice) ===")
    survivors = []
    # Group configs by (symbol, timeframe) to load data once per group.
    by_sym_tf: dict[tuple[str, str], list[dict]] = {}
    for cfg in configs:
        key = (cfg["symbol"], cfg["timeframe"])
        by_sym_tf.setdefault(key, []).append(cfg)

    t0 = time.time()
    n_total = len(configs)
    n_processed = 0
    for (sym, tf), group in by_sym_tf.items():
        df_full = load_data(sym, tf)
        is_df, _ = split_is_oos(df_full, 0.70)
        # Take first 6 months of IS for fast filter
        slice_end = is_df.index.min() + pd.Timedelta(days=180)
        df_slice = is_df.loc[:slice_end]
        print(f"  [{sym} {tf}] slice: {len(df_slice)} bars, {len(group)} configs")

        for cfg in group:
            ok, stats = pass1_fast_filter(df_slice, cfg, min_trades=30)
            cfg["_pass1"] = stats
            if ok:
                survivors.append(cfg)
            n_processed += 1
            if n_processed % log_every == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed
                eta = (n_total - n_processed) / rate if rate > 0 else 0
                print(f"    processed {n_processed}/{n_total}  survivors={len(survivors)}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s")
    print(f"  Pass 1 complete: {len(survivors)}/{n_total} survived ({100*len(survivors)/n_total:.1f}%)")
    PASS1_OUT.write_text(json.dumps(survivors, indent=2))
    return survivors


def run_pass2(pass1_survivors: list[dict], log_every: int = 25) -> list[dict]:
    """Pass 2: full event-driven backtest on the full 70% IS window.
    Keep configs with >= 200 closed trades in IS."""
    print("\n=== Pass 2: full event-driven backtest on full IS (70%) ===")
    by_sym_tf: dict[tuple[str, str], list[dict]] = {}
    for cfg in pass1_survivors:
        key = (cfg["symbol"], cfg["timeframe"])
        by_sym_tf.setdefault(key, []).append(cfg)

    survivors = []
    costs = Costs()
    t0 = time.time()
    n_total = len(pass1_survivors)
    n_processed = 0

    for (sym, tf), group in by_sym_tf.items():
        df_full = load_data(sym, tf)
        is_df, _ = split_is_oos(df_full, 0.70)
        print(f"  [{sym} {tf}] IS slice: {len(is_df)} bars, {len(group)} configs")

        for cfg in group:
            try:
                result = backtest(is_df, cfg, costs)
                metrics = compute_metrics(result, "default")
                cfg["_pass2_is_metrics"] = metrics
                # Keep if >= 200 trades AND net profit positive (small pre-filter
                # before we spend OOS effort on it).
                if metrics["n_closed"] >= 200 and metrics["net_profit_pct"] > 0:
                    survivors.append(cfg)
            except Exception as e:
                cfg["_pass2_error"] = str(e)
            n_processed += 1
            if n_processed % log_every == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed if elapsed > 0 else 0
                eta = (n_total - n_processed) / rate if rate > 0 else 0
                print(f"    processed {n_processed}/{n_total}  survivors={len(survivors)}  "
                      f"rate={rate:.2f}/s  eta={eta:.0f}s")

    print(f"  Pass 2 complete: {len(survivors)}/{n_total} survived ({100*len(survivors)/n_total:.1f}%)")
    PASS2_OUT.write_text(json.dumps(survivors, indent=2))
    return survivors


def run_pass3(pass2_survivors: list[dict], log_every: int = 10) -> list[dict]:
    """Pass 3: full backtest on untouched 30% OOS window. Measure, don't filter."""
    print("\n=== Pass 3: full backtest on untouched OOS (30%) ===")
    by_sym_tf: dict[tuple[str, str], list[dict]] = {}
    for cfg in pass2_survivors:
        key = (cfg["symbol"], cfg["timeframe"])
        by_sym_tf.setdefault(key, []).append(cfg)

    results = []
    costs = Costs()
    t0 = time.time()
    n_total = len(pass2_survivors)
    n_processed = 0

    for (sym, tf), group in by_sym_tf.items():
        df_full = load_data(sym, tf)
        _, oos_df = split_is_oos(df_full, 0.70)
        print(f"  [{sym} {tf}] OOS slice: {len(oos_df)} bars, {len(group)} configs")

        for cfg in group:
            try:
                result = backtest(oos_df, cfg, costs)
                metrics = compute_metrics(result, "default")
                cfg["_pass3_oos_metrics"] = metrics
                cfg["_pass3_oos_result"] = result  # keep for trade log export
                results.append(cfg)
            except Exception as e:
                cfg["_pass3_error"] = str(e)
            n_processed += 1
            if n_processed % log_every == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed if elapsed > 0 else 0
                eta = (n_total - n_processed) / rate if rate > 0 else 0
                print(f"    processed {n_processed}/{n_total}  rate={rate:.2f}/s  eta={eta:.0f}s")

    print(f"  Pass 3 complete: {len(results)}/{n_total} measured")
    # Don't write _pass3_oos_result (BacktestResult not JSON-serializable)
    out = []
    for cfg in results:
        c = {k: v for k, v in cfg.items() if not k.startswith("_pass3_oos_result")}
        out.append(c)
    PASS3_OUT.write_text(json.dumps(out, indent=2))
    return results


# ============================================================================
# results.csv writer
# ============================================================================

def write_results_csv(pass3_results: list[dict]) -> None:
    """Write results.csv with all metrics for IS and OOS, plus gate status."""
    print("\n=== Writing results.csv ===")
    rows = []
    for cfg in pass3_results:
        is_m = cfg.get("_pass2_is_metrics", {})
        oos_m = cfg.get("_pass3_oos_metrics", {})
        oos_gates = gate_check(oos_m, "OOS")
        is_gates = gate_check(is_m, "IS")

        row = {
            "id": cfg["id"],
            "symbol": cfg["symbol"],
            "timeframe": cfg["timeframe"],
            "side": cfg["side"],
            "trend_filter": cfg["trend_filter"]["kind"],
            "entry_trigger": cfg["entry_trigger"]["kind"],
            "confirmation": cfg["confirmation"]["kind"],
            "stop": cfg["stop"]["kind"],
            "target_r": cfg["target"]["r_mult"],
            "exit": cfg["exit"]["kind"],
            # IS metrics
            "is_n_closed": is_m.get("n_closed", 0),
            "is_win_rate": is_m.get("win_rate", 0),
            "is_avg_win_R": is_m.get("avg_win_R", 0),
            "is_avg_loss_R": is_m.get("avg_loss_R", 0),
            "is_pf": is_m.get("profit_factor", 0),
            "is_net_profit_pct": is_m.get("net_profit_pct", 0),
            "is_expectancy_R": is_m.get("expectancy_R", 0),
            "is_max_dd_pct": is_m.get("max_dd_pct", 0),
            "is_longest_loss_streak": is_m.get("longest_loss_streak", 0),
            "is_sharpe_daily": is_m.get("sharpe_daily", 0),
            # OOS metrics
            "oos_n_closed": oos_m.get("n_closed", 0),
            "oos_win_rate": oos_m.get("win_rate", 0),
            "oos_avg_win_R": oos_m.get("avg_win_R", 0),
            "oos_avg_loss_R": oos_m.get("avg_loss_R", 0),
            "oos_pf": oos_m.get("profit_factor", 0),
            "oos_net_profit_pct": oos_m.get("net_profit_pct", 0),
            "oos_expectancy_R": oos_m.get("expectancy_R", 0),
            "oos_max_dd_pct": oos_m.get("max_dd_pct", 0),
            "oos_longest_loss_streak": oos_m.get("longest_loss_streak", 0),
            "oos_sharpe_daily": oos_m.get("sharpe_daily", 0),
            # OOS-vs-IS comparisons
            "oos_win_rate_drop_pp": (is_m.get("win_rate", 0) - oos_m.get("win_rate", 0)) * 100,
            "oos_net_profit_stays_positive": int(oos_m.get("net_profit_pct", 0) > 0),
            # OOS gate checks
            "gate_oos_trades_pass": int(oos_gates["gate_trades"][0]),
            "gate_oos_win_rate_pass": int(oos_gates["gate_win_rate"][0]),
            "gate_oos_avg_win_R_pass": int(oos_gates["gate_avg_win_R"][0]),
            "gate_oos_pf_pass": int(oos_gates["gate_pf"][0]),
            "gate_oos_net_profit_pass": int(oos_gates["gate_net_profit"][0]),
            "gate_oos_expectancy_pass": int(oos_gates["gate_expectancy"][0]),
            "gate_oos_max_dd_pass": int(oos_gates["gate_max_dd"][0]),
            "gate_oos_all_pass": int(all(passed for passed, _, _ in oos_gates.values())),
            "gate_oos_win_drop_pass": int((is_m.get("win_rate", 0) - oos_m.get("win_rate", 0)) * 100 <= GATE_THRESHOLDS["max_oos_win_drop_pp"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by OOS net profit descending
    df = df.sort_values("oos_net_profit_pct", ascending=False).reset_index(drop=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"  Wrote {RESULTS_PATH} with {len(df)} rows")
    if len(df) > 0:
        print(f"  Top by OOS net profit:")
        for _, r in df.head(5).iterrows():
            print(f"    {r['id']}: OOS net={r['oos_net_profit_pct']*100:.1f}%  win={r['oos_win_rate']*100:.1f}%  "
                  f"trades={r['oos_n_closed']}  dd={r['oos_max_dd_pct']*100:.1f}%  all_gates={r['gate_oos_all_pass']}")


def main():
    configs = json.loads((DATA_DIR / "strategies.json").read_text())
    print(f"Loaded {len(configs)} configs")

    # Pass 1
    if PASS1_OUT.exists():
        pass1_survivors = json.loads(PASS1_OUT.read_text())
        # Verify they have the right structure
        if pass1_survivors and "_pass1" in pass1_survivors[0]:
            print(f"  [cache] Pass 1 already complete: {len(pass1_survivors)} survivors")
        else:
            pass1_survivors = run_pass1(configs)
    else:
        pass1_survivors = run_pass1(configs)

    # Pass 2
    if PASS2_OUT.exists():
        pass2_survivors = json.loads(PASS2_OUT.read_text())
        if pass2_survivors and "_pass2_is_metrics" in pass2_survivors[0]:
            print(f"  [cache] Pass 2 already complete: {len(pass2_survivors)} survivors")
        else:
            pass2_survivors = run_pass2(pass1_survivors)
    else:
        pass2_survivors = run_pass2(pass1_survivors)

    # Pass 3
    if PASS3_OUT.exists():
        cached = json.loads(PASS3_OUT.read_text())
        # We need to re-run Pass 3 because the cached version doesn't have
        # the BacktestResult object (not JSON-serializable). But for ranking
        # and gate checks, the cached metrics are sufficient.
        # Re-run only on the pass2 survivors to get BacktestResult objects.
        print(f"  [cache] Pass 3 metrics found, but re-running to get trade-level data for top candidates...")
        pass3_results = run_pass3(pass2_survivors)
    else:
        pass3_results = run_pass3(pass2_survivors)

    # Write results.csv
    write_results_csv(pass3_results)


if __name__ == "__main__":
    main()
