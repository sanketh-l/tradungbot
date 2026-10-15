"""
Phase 5: Robustness checks for top candidates.

For each of the top N candidates (by OOS net profit), run:
  1. Cost stress — double commission + slippage. Must remain net positive.
  2. Parameter sensitivity — perturb each numeric parameter ±10% and ±25%.
     Reject if performance collapses (defined as: net profit goes from positive
     to negative, OR win rate drops by >15pp from baseline).
  3. Walk-forward — split IS into 5 sequential sub-windows. Must be profitable
     (net positive) in >= 4 of 5.
  4. Regime split — evaluate separately on trending (ADX>25) and ranging
     (ADX<20) periods. Disclose if dependent on one regime.

Also exports:
  - trades_best.csv: full trade log for the top candidate
  - equity_curve.png: equity curve for the top candidate, IS and OOS overlaid
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester import Costs, BacktestResult, backtest
from evaluate import (
    load_data, split_is_oos, compute_metrics, gate_check, all_gates_passed,
    GATE_THRESHOLDS
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOWNLOAD = Path(__file__).resolve().parent.parent / "download"
DOWNLOAD.mkdir(parents=True, exist_ok=True)

ROBUSTNESS_OUT = DOWNLOAD / "robustness_results.csv"
TRADES_BEST_OUT = DOWNLOAD / "trades_best.csv"
EQUITY_CURVE_PNG = DOWNLOAD / "equity_curve.png"


# ============================================================================
# 1. Cost stress
# ============================================================================

def cost_stress(df_oos: pd.DataFrame, cfg: dict) -> dict:
    base = backtest(df_oos, cfg, Costs())
    base_m = compute_metrics(base, "default")
    stressed = backtest(df_oos, cfg, Costs().doubled())
    stressed_m = compute_metrics(stressed, "2x_costs")
    return {
        "base_net_profit_pct": base_m["net_profit_pct"],
        "stressed_net_profit_pct": stressed_m["net_profit_pct"],
        "base_win_rate": base_m["win_rate"],
        "stressed_win_rate": stressed_m["win_rate"],
        "remains_positive": stressed_m["net_profit_pct"] > 0,
    }


# ============================================================================
# 2. Parameter sensitivity
# ============================================================================

def _perturb(cfg: dict, factor: float) -> dict:
    """Return a copy of cfg with every numeric parameter multiplied by `factor`,
    rounded sensibly. Integer-valued params (periods, lengths) are rounded to
    int and clipped to >= 1."""
    cfg2 = copy.deepcopy(cfg)
    def walk(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                walk(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                if k in ("r_mult",):
                    new = round(v * factor, 2)
                elif k in ("mult", "pct", "max_dist_pct", "n_std", "trail_r"):
                    new = round(v * factor, 3)
                elif isinstance(v, int):
                    new = max(1, int(round(v * factor)))
                else:
                    new = round(v * factor, 3)
                d[k] = new
    walk(cfg2)
    return cfg2


def parameter_sensitivity(df_oos: pd.DataFrame, cfg: dict) -> dict:
    """Perturb each parameter ±10% and ±25% (whole-config perturbation).
    Report net profit and win rate for each perturbation."""
    base_m = compute_metrics(backtest(df_oos, cfg, Costs()), "default")
    results = {}
    for label, factor in [("-10%", 0.9), ("+10%", 1.1), ("-25%", 0.75), ("+25%", 1.25)]:
        try:
            cfg2 = _perturb(cfg, factor)
            m = compute_metrics(backtest(df_oos, cfg2, Costs()), f"perturbed_{label}")
            results[label] = {
                "net_profit_pct": m["net_profit_pct"],
                "win_rate": m["win_rate"],
                "n_closed": m["n_closed"],
            }
        except Exception as e:
            results[label] = {"error": str(e)}

    # "Collapses" = any perturbation flips net profit from positive to negative
    # OR drops win rate by >15pp from base
    base_np = base_m["net_profit_pct"]
    base_wr = base_m["win_rate"]
    collapse_flags = []
    for label, r in results.items():
        if "error" in r:
            collapse_flags.append(True)
            continue
        if base_np > 0 and r["net_profit_pct"] < 0:
            collapse_flags.append(True)
        elif (base_wr - r["win_rate"]) * 100 > 15:
            collapse_flags.append(True)
        else:
            collapse_flags.append(False)
    return {
        "base_net_profit_pct": base_np,
        "base_win_rate": base_wr,
        "perturbations": results,
        "any_collapse": any(collapse_flags),
    }


# ============================================================================
# 3. Walk-forward (5 windows on IS)
# ============================================================================

def walk_forward(df_is: pd.DataFrame, cfg: dict, n_windows: int = 5) -> dict:
    n = len(df_is)
    window_size = n // n_windows
    results = []
    for w in range(n_windows):
        start = w * window_size
        end = (w + 1) * window_size if w < n_windows - 1 else n
        df_w = df_is.iloc[start:end]
        try:
            m = compute_metrics(backtest(df_w, cfg, Costs()), f"wf_window_{w+1}")
            results.append({
                "window": w + 1,
                "start": str(df_w.index.min()),
                "end": str(df_w.index.max()),
                "n_closed": m["n_closed"],
                "net_profit_pct": m["net_profit_pct"],
                "win_rate": m["win_rate"],
            })
        except Exception as e:
            results.append({"window": w + 1, "error": str(e)})
    n_profitable = sum(1 for r in results if r.get("net_profit_pct", 0) > 0)
    return {
        "windows": results,
        "n_profitable_windows": n_profitable,
        "passes_walk_forward": n_profitable >= 4,
    }


# ============================================================================
# 4. Regime split
# ============================================================================

def regime_split(df_oos: pd.DataFrame, cfg: dict) -> dict:
    """Split bars by ADX regime (computed on the cfg's timeframe) and run
    backtest on each subset."""
    from backtester import adx
    adx_line = adx(df_oos, 14)
    trending_mask = (adx_line > 25).fillna(False)
    ranging_mask = (adx_line < 20).fillna(False)
    out = {}
    for label, mask in [("trending_adx_gt_25", trending_mask), ("ranging_adx_lt_20", ranging_mask)]:
        df_sub = df_oos[mask]
        if len(df_sub) < 100:
            out[label] = {"n_bars": int(len(df_sub)), "skipped": True}
            continue
        try:
            m = compute_metrics(backtest(df_sub, cfg, Costs()), label)
            out[label] = {
                "n_bars": int(len(df_sub)),
                "n_closed": m["n_closed"],
                "net_profit_pct": m["net_profit_pct"],
                "win_rate": m["win_rate"],
            }
        except Exception as e:
            out[label] = {"error": str(e), "n_bars": int(len(df_sub))}
    # "Dependent on one regime" = profitable in one and losing in the other
    t_np = out.get("trending_adx_gt_25", {}).get("net_profit_pct", 0)
    r_np = out.get("ranging_adx_lt_20", {}).get("net_profit_pct", 0)
    out["regime_dependent"] = bool(
        (t_np > 0 and r_np < 0) or (t_np < 0 and r_np > 0)
    )
    return out


# ============================================================================
# Trade log export
# ============================================================================

def export_trade_log(result: BacktestResult, path: Path) -> None:
    rows = []
    for t in result.trades:
        rows.append({
            "entry_time": t.entry_time,
            "exit_time":  t.exit_time,
            "side":       t.side,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "stop":        t.stop,
            "target":      t.target,
            "size":        t.size,
            "gross_pnl":   t.gross_pnl,
            "cost":        t.cost,
            "net_pnl":     t.net_pnl,
            "R_multiple":  t.r_multiple,
            "exit_reason": t.exit_reason,
            "bars_held":   t.bars_held,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Wrote {path} with {len(df)} trades")


# ============================================================================
# Equity curve plot
# ============================================================================

def plot_equity_curve(cfg: dict, df_is: pd.DataFrame, df_oos: pd.DataFrame,
                      path: Path) -> None:
    """Plot IS and OOS equity curves overlaid (with a marker at the IS/OOS
    boundary)."""
    # Try to use Noto Sans SC for CJK + DejaVu Sans for symbols (per spec rule 7)
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    is_res = backtest(df_is, cfg, Costs())
    oos_res = backtest(df_oos, cfg, Costs())

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # IS equity curve
    is_eq = is_res.equity_curve
    is_t = pd.to_datetime(is_res.equity_curve_times)
    ax.plot(is_t, is_eq, color="#1f77b4", linewidth=1.2, label=f"IS (70%)  final={is_res.final_equity:.3f}")

    # OOS equity curve — continue from where IS ended
    oos_eq = oos_res.equity_curve * is_res.final_equity  # scale to continue from IS end
    oos_t = pd.to_datetime(oos_res.equity_curve_times)
    ax.plot(oos_t, oos_eq, color="#d62728", linewidth=1.2, label=f"OOS (30%)  final={is_res.final_equity * oos_res.final_equity:.3f}")

    # Boundary marker
    boundary = df_is.index.max()
    ax.axvline(boundary, color="black", linestyle="--", alpha=0.5, label="IS/OOS boundary")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equity (USDT, starting at 1.0)")
    title = (f"Equity Curve — {cfg['id']}\n"
             f"{cfg['symbol']} {cfg['timeframe']} {cfg['side']} | "
             f"trend={cfg['trend_filter']['kind']} entry={cfg['entry_trigger']['kind']} "
             f"stop={cfg['stop']['kind']} target_R={cfg['target']['r_mult']} exit={cfg['exit']['kind']}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Wrote {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n=== Phase 5: Robustness checks on top candidates ===")

    # Load Pass 3 results (already includes _pass3_oos_metrics)
    pass3_path = DATA_DIR / "pass3_oos_results.json"
    pass3 = json.loads(pass3_path.read_text())

    # Sort by OOS net profit descending
    pass3_by_net = sorted(pass3, key=lambda c: c.get("_pass3_oos_metrics", {}).get("net_profit_pct", -1e9), reverse=True)
    # Also pick the top 5 that PASS the 200-trade gate (separate ranking)
    pass_200 = [c for c in pass3 if c.get("_pass3_oos_metrics", {}).get("n_closed", 0) >= 200]
    pass_200_by_net = sorted(pass_200, key=lambda c: c.get("_pass3_oos_metrics", {}).get("net_profit_pct", -1e9), reverse=True)

    top_n = pass3_by_net[:5]
    # Add the top configs that pass 200-trade gate (dedupe by id)
    seen_ids = {c["id"] for c in top_n}
    for c in pass_200_by_net[:5]:
        if c["id"] not in seen_ids:
            top_n.append(c)
            seen_ids.add(c["id"])

    print(f"  Top {len(top_n)} candidates selected (top-5 by OOS net profit + top configs with >=200 OOS trades):")
    for c in top_n:
        m = c["_pass3_oos_metrics"]
        print(f"    {c['id']}: OOS net={m['net_profit_pct']*100:.1f}%  win={m['win_rate']*100:.1f}%  "
              f"trades={m['n_closed']}  dd={m['max_dd_pct']*100:.1f}%")

    # Run robustness for each
    all_results = []
    for cfg in top_n:
        print(f"\n  -- {cfg['id']} --")
        df_full = load_data(cfg["symbol"], cfg["timeframe"])
        df_is, df_oos = split_is_oos(df_full, 0.70)

        # Cost stress
        cs = cost_stress(df_oos, cfg)
        print(f"    cost_stress: base_net={cs['base_net_profit_pct']*100:.1f}%  "
              f"2x_net={cs['stressed_net_profit_pct']*100:.1f}%  remains_positive={cs['remains_positive']}")

        # Parameter sensitivity
        ps = parameter_sensitivity(df_oos, cfg)
        print(f"    param_sensitivity: any_collapse={ps['any_collapse']}")
        for label, r in ps["perturbations"].items():
            if "error" not in r:
                print(f"      {label}: net={r['net_profit_pct']*100:.1f}%  win={r['win_rate']*100:.1f}%  trades={r['n_closed']}")

        # Walk-forward
        wf = walk_forward(df_is, cfg, n_windows=5)
        print(f"    walk_forward: n_profitable_windows={wf['n_profitable_windows']}/5  passes={wf['passes_walk_forward']}")
        for w in wf["windows"]:
            if "error" not in w:
                print(f"      window {w['window']}: net={w['net_profit_pct']*100:.1f}%  win={w['win_rate']*100:.1f}%  trades={w['n_closed']}")

        # Regime split
        rs = regime_split(df_oos, cfg)
        print(f"    regime_split: regime_dependent={rs['regime_dependent']}")
        for label in ["trending_adx_gt_25", "ranging_adx_lt_20"]:
            r = rs.get(label, {})
            if "error" not in r and not r.get("skipped"):
                print(f"      {label}: bars={r['n_bars']}  net={r['net_profit_pct']*100:.1f}%  win={r['win_rate']*100:.1f}%  trades={r['n_closed']}")

        all_results.append({
            "id": cfg["id"],
            "symbol": cfg["symbol"], "timeframe": cfg["timeframe"], "side": cfg["side"],
            "trend_filter": cfg["trend_filter"]["kind"],
            "entry_trigger": cfg["entry_trigger"]["kind"],
            "confirmation": cfg["confirmation"]["kind"],
            "stop": cfg["stop"]["kind"],
            "target_r": cfg["target"]["r_mult"],
            "exit": cfg["exit"]["kind"],
            "oos_net_profit_pct": cfg["_pass3_oos_metrics"]["net_profit_pct"],
            "oos_win_rate": cfg["_pass3_oos_metrics"]["win_rate"],
            "oos_n_closed": cfg["_pass3_oos_metrics"]["n_closed"],
            "oos_max_dd_pct": cfg["_pass3_oos_metrics"]["max_dd_pct"],
            # Robustness outcomes
            "cost_stress_remains_positive": int(cs["remains_positive"]),
            "cost_stress_2x_net_pct": cs["stressed_net_profit_pct"],
            "param_sensitivity_any_collapse": int(ps["any_collapse"]),
            "walk_forward_n_profitable": wf["n_profitable_windows"],
            "walk_forward_passes": int(wf["passes_walk_forward"]),
            "regime_dependent": int(rs["regime_dependent"]),
            "regime_trending_net_pct": rs.get("trending_adx_gt_25", {}).get("net_profit_pct", None),
            "regime_ranging_net_pct":  rs.get("ranging_adx_lt_20", {}).get("net_profit_pct", None),
            "regime_trending_win_rate": rs.get("trending_adx_gt_25", {}).get("win_rate", None),
            "regime_ranging_win_rate":  rs.get("ranging_adx_lt_20", {}).get("win_rate", None),
            # Full per-perturbation detail stored as JSON string
            "param_sensitivity_detail": json.dumps(ps["perturbations"]),
            "walk_forward_detail": json.dumps(wf["windows"]),
            "regime_split_detail": json.dumps(rs),
        })

    df_r = pd.DataFrame(all_results)
    df_r.to_csv(ROBUSTNESS_OUT, index=False)
    print(f"\n  Wrote {ROBUSTNESS_OUT} with {len(df_r)} rows")

    # Pick "best" candidate for trade log + equity curve.
    # Preference order:
    #   1. Highest OOS net profit among those with >=200 OOS trades (passes trade-count gate)
    #   2. If none, highest OOS net profit overall
    eligible = [c for c in top_n if c["_pass3_oos_metrics"]["n_closed"] >= 200]
    if eligible:
        eligible.sort(key=lambda c: c["_pass3_oos_metrics"]["net_profit_pct"], reverse=True)
        best = eligible[0]
        print(f"\n  Best candidate for trade log + equity curve: {best['id']}  (>=200 OOS trades)")
    else:
        best = top_n[0]
        print(f"\n  Best candidate for trade log + equity curve: {best['id']}  (<200 OOS trades — note this)")
        print(f"    (No candidate has >=200 OOS trades among top-by-net-profit; using top by net profit)")

    df_full = load_data(best["symbol"], best["timeframe"])
    df_is, df_oos = split_is_oos(df_full, 0.70)
    oos_result = backtest(df_oos, best, Costs())
    export_trade_log(oos_result, TRADES_BEST_OUT)
    plot_equity_curve(best, df_is, df_oos, EQUITY_CURVE_PNG)

    # Also save the best config to disk for reproducibility
    (DATA_DIR / "best_config.json").write_text(json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    main()
