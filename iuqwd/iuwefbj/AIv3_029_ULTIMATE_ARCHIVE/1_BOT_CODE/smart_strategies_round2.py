"""
Round 2 of smart strategies — applying what Round 1 taught us.

Lessons from Round 1:
  1. TIGHT STOPS are the lever. Strategy G (5-bar low stop in low-vol) was the
     only profitable strategy. ATR-based stops in high-vol regimes made 2R
     targets unreachable.
  2. "Stacking" oversold signals didn't help — they're correlated, not
     independent, so joint probability stayed near 30%.
  3. Momentum (breakout) beat reversion (oversold) — by a small margin.

Round 2 design principles:
  - Use the SIGNAL BAR's low as the stop reference (just 0.05*ATR below it).
    This makes the stop genuinely tight — risk per trade is typically 0.2-0.4%.
  - Force support/resistance CONFLUENCE: only enter if the signal bar's low is
    near (within 0.3%) of a known level (EMA, VWAP, prior swing).
  - Lean into momentum variants (Strategy G family was best in Round 1).
  - Add an asymmetric idea: enter on the SECOND bar of a confirmed reversal
    (not the first) to reduce false signals.
  - Add a HTF-strict filter: only trade when 4h trend is STRONGLY aligned
    (4h ADX > 25 AND 4h EMA50 > EMA200).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backtester import (
    Costs, BacktestResult, Trade,
    ema, rsi, atr, adx, bollinger, vwap_rolling, donchian
)
from smart_strategies import (
    zscore, resample_htf, taker_buy_ratio, reindex_htf_to_ltf,
    htf_trend_up, htf_trend_dn,
    backtest_smart, compute_metrics, all_gates_passed,
    load_1h_with_htf, split_is_oos,
    SmartStrategy, make_stop_from_recent_low, make_stop_from_5bar_low,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOWNLOAD = Path(__file__).resolve().parent.parent / "download"
OUT_PATH = DOWNLOAD / "smart_strategies_round2_results.csv"


# ============================================================================
# New tighter stop functions
# ============================================================================

def make_stop_signal_bar_low(df: pd.DataFrame, atr_buffer_mult: float = 0.05) -> pd.Series:
    """Stop = signal bar's low - 0.05 * ATR(14). VERY tight.
    For shorts: stop = signal bar's high + 0.05 * ATR(14). (Handled by caller.)
    """
    return df["low"].shift(0) - atr_buffer_mult * atr(df, 14)


def make_stop_signal_bar_high(df: pd.DataFrame, atr_buffer_mult: float = 0.05) -> pd.Series:
    """Stop for shorts = signal bar's high + 0.05 * ATR(14)."""
    return df["high"].shift(0) + atr_buffer_mult * atr(df, 14)


def make_stop_prev_bar_low(df: pd.DataFrame, atr_buffer_mult: float = 0.1) -> pd.Series:
    """Stop = previous bar's low - 0.1 * ATR(14). Tight but with a tiny buffer."""
    return df["low"].shift(1) - atr_buffer_mult * atr(df, 14)


# ============================================================================
# HTF filters (stricter)
# ============================================================================

def htf_strong_uptrend(htf: pd.DataFrame) -> pd.Series:
    """4h strongly bullish: ADX(14) > 25 AND EMA50 > EMA200."""
    return (adx(htf, 14) > 25) & (ema(htf["close"], 50) > ema(htf["close"], 200))


def htf_strong_downtrend(htf: pd.DataFrame) -> pd.Series:
    return (adx(htf, 14) > 25) & (ema(htf["close"], 50) < ema(htf["close"], 200))


# ============================================================================
# Round 2 strategies
# ============================================================================

def strat_R2G1_low_vol_breakout_tightstop(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """R2-G1: Low-vol breakout with signal-bar-low stop (tightest possible).

    Same as Round 1 Strategy G, but now with the signal-bar-low stop.
    Thesis: low-vol regime + 20-bar high breakout + tight stop = small risk,
    modest 2R target achievable on follow-through.
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    atr_norm = atr(df, 14) / df["close"]
    low_vol = atr_norm < 0.010  # slightly looser than Round 1's 0.008
    upper_20 = df["high"].rolling(20).max().shift(1)
    breakout = df["close"] > upper_20
    vol_surge = df["volume"] > 1.3 * df["volume"].rolling(20).mean()
    return htf_strong_up & low_vol & breakout & vol_surge


def strat_R2G2_breakout_with_ema_align(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """R2-G2: Breakout above 20-bar high, with 1h EMA20 > EMA50 > EMA200 (stacked).

    Thesis: in a fully stacked 1h trend, breakouts of consolidation have
    high follow-through. Tight stop at signal bar low.
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    ema20 = ema(df["close"], 20); ema50 = ema(df["close"], 50); ema200 = ema(df["close"], 200)
    stacked = (ema20 > ema50) & (ema50 > ema200)
    upper_20 = df["high"].rolling(20).max().shift(1)
    breakout = df["close"] > upper_20
    return htf_strong_up & stacked & breakout


def strat_R2G3_breakout_pullback(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Signal:
    """R2-G3: Breakout then pullback to 10-EMA, enter on the reclaim.

    Thesis: classic breakout-pullback setup. Initial breakout establishes
    new support; pullback to fast EMA gives low-risk entry with tight stop
    at pullback low.
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    upper_20 = df["high"].rolling(20).max().shift(1)
    # Did we break out in the last 5 bars?
    broke_recent = ((df["close"].shift(1) > upper_20.shift(1)) |
                    (df["close"].shift(2) > upper_20.shift(2)) |
                    (df["close"].shift(3) > upper_20.shift(3)) |
                    (df["close"].shift(4) > upper_20.shift(4)) |
                    (df["close"].shift(5) > upper_20.shift(5)))
    # Pullback to 10-EMA
    ema10 = ema(df["close"], 10)
    pullback = (df["low"] <= ema10 * 1.003) & (df["close"] > ema10)
    # Reclaim: close back above ema10
    return htf_strong_up & broke_recent & pullback


def strat_R2G4_vwap_breakout_tight(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """R2-G4: Breakout above VWAP(50) with very tight stop.

    Thesis: VWAP reclaim after recent consolidation, with tight stop at
    signal bar low.
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    v = vwap_rolling(df, 50)
    # Was below VWAP 1-3 bars ago, now above
    was_below = (df["close"].shift(1) < v.shift(1)) | \
                (df["close"].shift(2) < v.shift(2)) | \
                (df["close"].shift(3) < v.shift(3))
    reclaim = df["close"] > v
    vol_surge = df["volume"] > 1.5 * df["volume"].rolling(20).mean()
    return htf_strong_up & was_below & reclaim & vol_surge


def strat_R2G5_donchian_breakout_strict(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """R2-G5: Donchian(20) breakout with strict HTF filter and tight stop.

    Thesis: classic turtle breakout, but only in strong 4h trends, with tight
    stop at signal bar low.
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    upper_20 = df["high"].rolling(20).max().shift(1)
    breakout = df["close"] > upper_20
    # New 20-bar high
    is_new_high = df["close"] == df["close"].rolling(20).max()
    return htf_strong_up & breakout & is_new_high


def strat_R2R1_pin_at_vwap(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """R2-R1: Bullish pin bar AT VWAP (not EMA) in strong uptrend.

    Thesis: VWAP is a stronger support than EMA in crypto. Pin bar at VWAP
    with tight stop at pin low.
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    op = df["open"]; cl = df["close"]; hi = df["high"]; lo = df["low"]
    body = (cl - op).abs()
    rng = (hi - lo).replace(0, np.nan)
    lower_wick = np.minimum(cl, op) - lo
    is_pin = (lower_wick > 2 * body) & (lower_wick > 0.55 * rng) & (cl > op)
    v = vwap_rolling(df, 50)
    at_vwap = (lo <= v * 1.003) & (cl > v * 0.997)
    return htf_strong_up & is_pin & at_vwap


def strat_R2R2_two_bar_reversal_at_ema200(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """R2-R2: Two-bar reversal (down bar followed by up bar that engulfs it) at EMA200.

    Thesis: EMA200 is a major support/resistance level. A two-bar reversal
    there is a high-probability bounce. Tight stop at the low of the two bars.
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    op = df["open"]; cl = df["close"]; hi = df["high"]; lo = df["low"]
    # Bar -1 was a down bar
    down_bar = cl.shift(1) < op.shift(1)
    # Bar 0 is an up bar that engulfs bar -1's range
    up_bar = cl > op
    engulf = (cl > hi.shift(1)) & (op < lo.shift(1))
    ema200 = ema(cl, 200)
    at_ema200 = (lo <= ema200 * 1.003) & (cl > ema200 * 0.997)
    return htf_strong_up & down_bar & up_bar & engulf & at_ema200


def strat_R2R3_oversold_spike_reversal(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """R2-R3: RSI(2) < 5 (very oversold on short-term) + bullish engulfing + HTF up.

    Thesis: RSI(2) < 5 is extreme short-term oversold. Combined with bullish
    engulfing (a reversal candle) and HTF uptrend, this is a high-probability
    bounce. Stop at engulfing bar low.
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    r2 = rsi(df["close"], 2)
    op = df["open"]; cl = df["close"]
    body_prev = cl.shift(1) - op.shift(1)
    body_curr = cl - op
    engulfing = (body_prev < 0) & (body_curr > 0) & \
                (cl > op.shift(1)) & (op < cl.shift(1))
    return htf_strong_up & (r2 < 5) & engulfing


def strat_R2M1_momentum_continuation(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """R2-M1: Strong momentum bar (close in top 25% of range, body > 60% of range)
    in HTF uptrend, enter on next bar with tight stop at momentum bar's low.

    Thesis: a momentum ignition bar often continues for 2-3 more bars. Tight
    stop at the momentum bar's low (which is usually close to its open).
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    op = df["open"]; cl = df["close"]; hi = df["high"]; lo = df["low"]
    rng = (hi - lo).replace(0, np.nan)
    body = cl - op
    body_ratio = body.abs() / rng
    close_position = (cl - lo) / rng  # 1.0 = close at high
    is_momentum = (body > 0) & (body_ratio > 0.6) & (close_position > 0.75) & \
                  (df["volume"] > 1.5 * df["volume"].rolling(20).mean())
    return htf_strong_up & is_momentum


def strat_R2M2_breakout_volume_explosion(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """R2-M2: 20-bar high breakout with 3x average volume + taker-buy > 0.55.

    Thesis: institutional-grade breakout signature. Tight stop at signal bar low.
    """
    htf_strong_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
    upper_20 = df["high"].rolling(20).max().shift(1)
    breakout = df["close"] > upper_20
    vol_explosion = df["volume"] > 3.0 * df["volume"].rolling(20).mean()
    buy_pressure = taker_buy_ratio(df) > 0.55
    return htf_strong_up & breakout & vol_explosion & buy_pressure


# Short mirrors for the best candidates
def strat_R2G1_short(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    htf_strong_dn = reindex_htf_to_ltf(htf_strong_downtrend(htf), df.index)
    atr_norm = atr(df, 14) / df["close"]
    low_vol = atr_norm < 0.010
    lower_20 = df["low"].rolling(20).min().shift(1)
    breakout = df["close"] < lower_20
    vol_surge = df["volume"] > 1.3 * df["volume"].rolling(20).mean()
    return htf_strong_dn & low_vol & breakout & vol_surge


def strat_R2M2_short(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    htf_strong_dn = reindex_htf_to_ltf(htf_strong_downtrend(htf), df.index)
    lower_20 = df["low"].rolling(20).min().shift(1)
    breakout = df["close"] < lower_20
    vol_explosion = df["volume"] > 3.0 * df["volume"].rolling(20).mean()
    sell_pressure = taker_buy_ratio(df) < 0.45
    return htf_strong_dn & breakout & vol_explosion & sell_pressure


def strat_R2R3_short(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    htf_strong_dn = reindex_htf_to_ltf(htf_strong_downtrend(htf), df.index)
    r2 = rsi(df["close"], 2)
    op = df["open"]; cl = df["close"]
    body_prev = cl.shift(1) - op.shift(1)
    body_curr = cl - op
    engulfing = (body_prev > 0) & (body_curr < 0) & \
                (cl < op.shift(1)) & (op > cl.shift(1))
    return htf_strong_dn & (r2 > 95) & engulfing


# ============================================================================
# Strategy registry — Round 2
# ============================================================================

def for_long(stop_fn):
    """Wrap a stop function for long side (uses low)."""
    return stop_fn

def for_short(stop_fn_high):
    """Wrap a stop function for short side (uses high)."""
    return stop_fn_high


ROUND2_STRATEGIES = [
    # Momentum / breakout family (worked best in Round 1)
    SmartStrategy("R2G1_lowvol_break_tight",  "long",  strat_R2G1_low_vol_breakout_tightstop,
                  make_stop_signal_bar_low, 2.0, 24,
                  "Low-vol breakout with signal-bar-low stop. Tightest possible."),
    SmartStrategy("R2G2_breakout_stacked",   "long",  strat_R2G2_breakout_with_ema_align,
                  make_stop_signal_bar_low, 2.0, 24,
                  "Breakout in fully stacked 1h + 4h trend."),
    SmartStrategy("R2G3_breakout_pullback",  "long",  strat_R2G3_breakout_pullback,
                  make_stop_signal_bar_low, 2.0, 24,
                  "Breakout then pullback to 10-EMA, enter on reclaim."),
    SmartStrategy("R2G4_vwap_breakout",      "long",  strat_R2G4_vwap_breakout_tight,
                  make_stop_signal_bar_low, 2.0, 24,
                  "VWAP reclaim with volume, tight stop."),
    SmartStrategy("R2G5_donchian_strict",    "long",  strat_R2G5_donchian_breakout_strict,
                  make_stop_signal_bar_low, 2.0, 24,
                  "Donchian(20) breakout in strong 4h trend."),
    # Reversal family with tighter stops
    SmartStrategy("R2R1_pin_at_vwap",        "long",  strat_R2R1_pin_at_vwap,
                  make_stop_signal_bar_low, 2.0, 24,
                  "Bullish pin bar AT VWAP in strong uptrend."),
    SmartStrategy("R2R2_2bar_at_ema200",     "long",  strat_R2R2_two_bar_reversal_at_ema200,
                  make_stop_prev_bar_low, 2.0, 24,
                  "Two-bar engulfing reversal at EMA200 in uptrend."),
    SmartStrategy("R2R3_rsi2_oversold",      "long",  strat_R2R3_oversold_spike_reversal,
                  make_stop_signal_bar_low, 2.0, 24,
                  "RSI(2) < 5 + bullish engulfing in uptrend."),
    # New momentum ideas
    SmartStrategy("R2M1_momentum_bar",       "long",  strat_R2M1_momentum_continuation,
                  make_stop_signal_bar_low, 2.0, 12,
                  "Momentum ignition bar continuation, tight stop, short holding."),
    SmartStrategy("R2M2_breakout_volboom",   "long",  strat_R2M2_breakout_volume_explosion,
                  make_stop_signal_bar_low, 2.0, 24,
                  "Breakout + 3x volume + taker-buy > 0.55."),
    # Short mirrors
    SmartStrategy("R2G1_short",              "short", strat_R2G1_short,
                  make_stop_signal_bar_high, 2.0, 24,
                  "Short mirror of R2G1."),
    SmartStrategy("R2M2_short",              "short", strat_R2M2_short,
                  make_stop_signal_bar_high, 2.0, 24,
                  "Short mirror of R2M2."),
    SmartStrategy("R2R3_short",              "short", strat_R2R3_short,
                  make_stop_signal_bar_high, 2.0, 24,
                  "Short mirror of R2R3."),
]


def run_round2():
    print("\n" + "="*78)
    print("ROUND 2 — Smart strategies with tighter stops and stricter filters")
    print("="*78)

    all_results = []
    for sym in ["BTCUSDT", "ETHUSDT"]:
        print(f"\n>>> {sym}")
        df, htf = load_1h_with_htf(sym)
        df_is, df_oos = split_is_oos(df, 0.70)
        htf_is, htf_oos = split_is_oos(htf, 0.70)

        for strat in ROUND2_STRATEGIES:
            t0 = time.time()
            try:
                sig_is = strat.signal_fn(df_is, htf_is)
                stop_is = strat.stop_fn(df_is)
                sig_oos = strat.signal_fn(df_oos, htf_oos)
                stop_oos = strat.stop_fn(df_oos)

                is_result = backtest_smart(df_is, sig_is, strat.side, stop_is,
                                           r_mult=strat.r_mult, max_bars=strat.max_bars)
                oos_result = backtest_smart(df_oos, sig_oos, strat.side, stop_oos,
                                            r_mult=strat.r_mult, max_bars=strat.max_bars)
                is_m = compute_metrics(is_result)
                oos_m = compute_metrics(oos_result)

                row = {
                    "symbol": sym, "strategy": strat.name, "side": strat.side,
                    "thesis": strat.thesis,
                    "is_n_closed": is_m["n_closed"],
                    "is_win_rate": is_m["win_rate"],
                    "is_avg_win_R": is_m["avg_win_R"],
                    "is_pf": is_m["profit_factor"],
                    "is_net_profit_pct": is_m["net_profit_pct"],
                    "is_max_dd_pct": is_m["max_dd_pct"],
                    "is_expectancy_R": is_m["expectancy_R"],
                    "oos_n_closed": oos_m["n_closed"],
                    "oos_win_rate": oos_m["win_rate"],
                    "oos_avg_win_R": oos_m["avg_win_R"],
                    "oos_avg_loss_R": oos_m["avg_loss_R"],
                    "oos_pf": oos_m["profit_factor"],
                    "oos_net_profit_pct": oos_m["net_profit_pct"],
                    "oos_max_dd_pct": oos_m["max_dd_pct"],
                    "oos_expectancy_R": oos_m["expectancy_R"],
                    "oos_longest_loss_streak": oos_m["longest_loss_streak"],
                    "oos_sharpe_daily": oos_m["sharpe_daily"],
                    "all_oos_gates_pass": int(all_gates_passed(oos_m)),
                }
                all_results.append(row)
                dt = time.time() - t0
                tag = "✓" if row["all_oos_gates_pass"] else "✗"
                print(f"  {tag} {sym} {strat.name:30s}  IS: {is_m['n_closed']:>4}t wr={is_m['win_rate']*100:>5.1f}% net={is_m['net_profit_pct']*100:>+6.1f}%  "
                      f"OOS: {oos_m['n_closed']:>4}t wr={oos_m['win_rate']*100:>5.1f}% net={oos_m['net_profit_pct']*100:>+6.1f}% dd={oos_m['max_dd_pct']*100:>5.1f}%  ({dt:.1f}s)")
            except Exception as e:
                import traceback
                print(f"  ✗ {sym} {strat.name} FAILED: {e}")
                traceback.print_exc()

    df_r = pd.DataFrame(all_results)
    df_r.to_csv(OUT_PATH, index=False)
    print(f"\n>>> Wrote {OUT_PATH} with {len(df_r)} rows")

    # Show top 5 by OOS win rate
    print("\n=== Top 5 by OOS win rate ===")
    top = df_r.sort_values("oos_win_rate", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"  {r['symbol']} {r['strategy']:30s}  OOS: wr={r['oos_win_rate']*100:>5.1f}%  trades={r['oos_n_closed']:>4}  net={r['oos_net_profit_pct']*100:>+6.1f}%  dd={r['oos_max_dd_pct']*100:>5.1f}%  pf={r['oos_pf']:.2f}")

    print("\n=== Top 5 by OOS net profit ===")
    top = df_r.sort_values("oos_net_profit_pct", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"  {r['symbol']} {r['strategy']:30s}  OOS: wr={r['oos_win_rate']*100:>5.1f}%  trades={r['oos_n_closed']:>4}  net={r['oos_net_profit_pct']*100:>+6.1f}%  dd={r['oos_max_dd_pct']*100:>5.1f}%  pf={r['oos_pf']:.2f}")


if __name__ == "__main__":
    run_round2()
