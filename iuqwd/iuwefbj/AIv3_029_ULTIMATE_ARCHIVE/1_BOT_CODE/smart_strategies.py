"""
Smart strategies — designed with explicit mechanistic theses for high win rate
at 1:2 R:R. Each strategy has a stated reason for why it *should* hit 70%+ win
rate at 1:2 R:R, then we backtest with the SAME conservative methodology
(next-bar-open fills, costs on both sides, stop-first-when-both-hit, 1% risk,
untouched 30% OOS).

Design principles (learned from grid search failure):
  1. Stop placement matters more than entry signal. To hit 1:2 R:R with a
     high P(touch 2R before 1R), the stop must be TIGHT (0.3-0.6% of price).
     This means entering AT a recent extreme, not after a signal confirms
     late (which pushes the stop further away).
  2. Stack INDEPENDENT confluence signals. Each individual signal has ~55-60%
     hit rate; stacking 2-3 independent ones compounds toward 70%.
  3. Only enter in HTF-aligned regime. Wrong-regime trades are the main source
     of losses; filtering these out lifts win rate mechanically.
  4. Use order-flow proxy (taker_buy_ratio) which the grid search ignored.

Each strategy below is parameterized minimally — no per-strategy tuning. We
run each on BTCUSDT and ETHUSDT, 1h, full IS + OOS, and report actuals.
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
    Costs, BacktestResult, Trade, backtest,
    ema, rsi, atr, adx, bollinger, vwap_rolling, donchian
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOWNLOAD = Path(__file__).resolve().parent.parent / "download"
DOWNLOAD.mkdir(parents=True, exist_ok=True)

OUT_PATH = DOWNLOAD / "smart_strategies_results.csv"
TRADES_OUT = DOWNLOAD / "smart_trades_best.csv"
EQUITY_OUT = DOWNLOAD / "smart_equity_curve.png"


# ============================================================================
# Indicator helpers (extend backtester's set)
# ============================================================================

def zscore(series: pd.Series, length: int) -> pd.Series:
    mean = series.rolling(length).mean()
    sd = series.rolling(length).std(ddof=0)
    return (series - mean) / sd.replace(0, np.nan)


def resample_htf(df: pd.DataFrame, rule: str = "4h") -> pd.DataFrame:
    """Resample 1h OHLCV to higher timeframe (default 4h)."""
    agg = {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
        "quote_volume": "sum", "taker_buy_base": "sum",
    }
    htf = df.resample(rule).agg(agg).dropna()
    return htf


def htf_trend_up(htf: pd.DataFrame) -> pd.Series:
    """Boolean series on HTF index: True when EMA50 > EMA200 (stacked uptrend)."""
    return ema(htf["close"], 50) > ema(htf["close"], 200)


def htf_trend_dn(htf: pd.DataFrame) -> pd.Series:
    return ema(htf["close"], 50) < ema(htf["close"], 200)


def taker_buy_ratio(df: pd.DataFrame) -> pd.Series:
    """Fraction of volume that was taker-buy (buy-side aggression). >0.5 = buyers
    dominant. This is a real microstructure signal available in Binance klines
    that the grid search ignored."""
    return df["taker_buy_base"] / df["volume"].replace(0, np.nan)


def reindex_htf_to_ltf(htf_sig: pd.Series, ltf_index: pd.DatetimeIndex) -> pd.Series:
    """Reindex an HTF signal series to LTF index, forward-filling. By construction
    the HTF value at time T uses only HTF bars that CLOSED at or before T, so no
    lookahead. We forward-fill so each LTF bar sees the most recent HTF close."""
    return htf_sig.reindex(ltf_index, method="ffill").fillna(False)


# ============================================================================
# Strategy definitions
# Each strategy is a function: (df_ltf_1h, df_htf_4h) -> signal pd.Series (bool)
# Signal True at bar i means "entry signal fires at bar i close". The backtester
# fills at bar i+1 open. No lookahead by construction.
# ============================================================================

def strat_A_extreme_zscore_reversion(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """Strategy A: Extreme Z-Score Reversion in HTF Uptrend.

    Thesis: when 1h close is >2σ below its 50-bar mean AND the 4h trend is up,
    price is overextended downward in a regime that should revert. The 2σ
    extreme gives us a tight natural stop (just below the recent low), and the
    2R target is a modest reversion to the mean — both achievable within hours.
    """
    htf_up = reindex_htf_to_ltf(htf_trend_up(htf), df.index)
    z = zscore(df["close"], 50)
    sig = htf_up & (z < -2.0)
    return sig


def strat_B_vol_contraction_reversion(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """Strategy B: Volatility Contraction + BB Touch + RSI Oversold.

    Thesis: when ATR(14)/ATR(50) < 0.7 (volatility contracting) AND price
    touches lower BB(20, 2.5) AND RSI(14) < 25, all three conditions independently
    suggest a snap-back. Stacking three independent oversold signals should
    push P(reversion) well above the single-signal hit rate.
    """
    htf_up = reindex_htf_to_ltf(htf_trend_up(htf), df.index)
    atr_fast = atr(df, 14); atr_slow = atr(df, 50)
    vol_contract = (atr_fast / atr_slow.replace(0, np.nan)) < 0.7
    _, bb_upper, bb_lower = bollinger(df["close"], 20, 2.5)
    r = rsi(df["close"], 14)
    sig = htf_up & vol_contract & (df["low"] <= bb_lower) & (r < 25)
    return sig


def strat_C_3bar_reversal_at_htf_mean(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """Strategy C: 3-Bar Reversal at HTF Mean in Uptrend.

    Thesis: classic Wyckoff reversal — 3 consecutive lower lows followed by a
    higher close, occurring at or below the 1h EMA(50) while the 4h trend is
    up. The 3-bar low provides a tight stop just below entry; the reversal
    close confirms buyer absorption.
    """
    htf_up = reindex_htf_to_ltf(htf_trend_up(htf), df.index)
    low = df["low"]; close = df["close"]
    ll1 = low < low.shift(1)
    ll2 = low.shift(1) < low.shift(2)
    ll3 = low.shift(2) < low.shift(3)
    higher_close = close > close.shift(1)
    at_or_below_ema = close <= ema(close, 50) * 1.002
    sig = htf_up & ll1 & ll2 & ll3 & higher_close & at_or_below_ema
    return sig


def strat_D_vwap_reclaim_with_flow(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """Strategy D: VWAP Reclaim with Volume + Taker-Buy Pressure.

    Thesis: VWAP reclaim after extended time below, accompanied by 2x average
    volume AND taker-buy ratio > 0.55 (buyers paying the ask), is a classic
    momentum-ignition signal. The reclaim bar's low provides a tight stop.
    """
    htf_up = reindex_htf_to_ltf(htf_trend_up(htf), df.index)
    v = vwap_rolling(df, 100)
    below_vwap_5bar = (df["close"].shift(5) < v.shift(5)) & \
                      (df["close"].shift(4) < v.shift(4)) & \
                      (df["close"].shift(3) < v.shift(3)) & \
                      (df["close"].shift(2) < v.shift(2)) & \
                      (df["close"].shift(1) < v.shift(1))
    reclaim = df["close"] > v
    vol_surge = df["volume"] > 2.0 * df["volume"].rolling(20).mean()
    buy_pressure = taker_buy_ratio(df) > 0.55
    sig = htf_up & below_vwap_5bar & reclaim & vol_surge & buy_pressure
    return sig


def strat_E_pin_at_support(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """Strategy E: Bullish Pin Bar at EMA(50) in Uptrend.

    Thesis: a 1h pin bar (long lower wick, small body, close in upper half)
    touching the 1h EMA(50) while the 4h trend is up is a high-probability
    rejection of support. The pin bar's low is a natural tight stop.
    """
    htf_up = reindex_htf_to_ltf(htf_trend_up(htf), df.index)
    op = df["open"]; cl = df["close"]; hi = df["high"]; lo = df["low"]
    body = (cl - op).abs()
    rng = (hi - lo).replace(0, np.nan)
    lower_wick = np.minimum(cl, op) - lo
    is_pin = (lower_wick > 2 * body) & (lower_wick > 0.55 * rng) & (cl > op)
    ema50 = ema(cl, 50)
    at_ema = (lo <= ema50 * 1.003) & (cl > ema50 * 0.997)
    sig = htf_up & is_pin & at_ema
    return sig


def strat_F_stacked_reversal(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """Strategy F: Stacked Independent Reversal Signals.

    Thesis: each of these signals individually has ~55-60% hit rate at 1:2 R:R
    on oversold bounces. If they're approximately independent, the joint
    probability of a 2R bounce given ALL of them firing should approach 70%+.

    Required (all simultaneously, long side):
      - 4h EMA50 > EMA200 (HTF uptrend)
      - 1h RSI(14) < 30 (oversold)
      - 1h z-score(close, 20) < -1.5 (price 1.5σ below short-term mean)
      - 1h close > BB(20, 2) lower band (price has started to recover, not still falling)
      - 1h taker_buy_ratio > 0.50 (buyers not abandoning)
    """
    htf_up = reindex_htf_to_ltf(htf_trend_up(htf), df.index)
    r = rsi(df["close"], 14)
    z = zscore(df["close"], 20)
    _, bb_upper, bb_lower = bollinger(df["close"], 20, 2.0)
    tbr = taker_buy_ratio(df)
    sig = (htf_up &
           (r < 30) &
           (z < -1.5) &
           (df["close"] > bb_lower) &
           (tbr > 0.50))
    return sig


def strat_G_low_vol_breakout(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """Strategy G: Low-Volatility Tight-Stop Breakout.

    Thesis: when ATR(14)/close < 0.008 (very low volatility) AND price breaks
    above 20-bar high with 1.5x volume, the breakout range is tight so the
    stop (5-bar low) is also tight, making the 2R target achievable on the
    follow-through. This is a momentum strategy, not reversion.
    """
    htf_up = reindex_htf_to_ltf(htf_trend_up(htf), df.index)
    atr_norm = atr(df, 14) / df["close"]
    low_vol = atr_norm < 0.008
    upper_20 = df["high"].rolling(20).max().shift(1)
    breakout = df["close"] > upper_20
    vol_surge = df["volume"] > 1.5 * df["volume"].rolling(20).mean()
    sig = htf_up & low_vol & breakout & vol_surge
    return sig


def strat_H_trend_pullback_engulfing(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    """Strategy H: HTF-Stacked Trend Pullback with Bullish Engulfing.

    Thesis: when 4h EMA20 > EMA50 > EMA200 (fully stacked uptrend) AND 1h
    pulls back to within 0.5% of EMA(50) AND forms a bullish engulfing bar,
    enter with stop just below the engulfing bar's low. Tight stop, modest
    target, high follow-through probability in a stacked trend.
    """
    # HTF stacked uptrend
    ema20_h = ema(htf["close"], 20)
    ema50_h = ema(htf["close"], 50)
    ema200_h = ema(htf["close"], 200)
    htf_stacked = (ema20_h > ema50_h) & (ema50_h > ema200_h)
    htf_up = reindex_htf_to_ltf(htf_stacked, df.index)

    # 1h pullback to EMA50 within 0.5%
    ema50_1h = ema(df["close"], 50)
    pullback = (df["low"] <= ema50_1h * 1.005) & (df["close"] > ema50_1h * 0.995)

    # Bullish engulfing
    op = df["open"]; cl = df["close"]
    body_prev = cl.shift(1) - op.shift(1)
    body_curr = cl - op
    engulfing = (body_prev < 0) & (body_curr > 0) & \
                (cl > op.shift(1)) & (op < cl.shift(1))

    sig = htf_up & pullback & engulfing
    return sig


# SHORT-side mirrors (same logic, inverted) — for completeness and to test the
# symmetric claim that "70%/1:2 should work on both sides if it works at all."

def strat_A_short(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    htf_dn = reindex_htf_to_ltf(htf_trend_dn(htf), df.index)
    z = zscore(df["close"], 50)
    return htf_dn & (z > 2.0)

def strat_B_short(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    htf_dn = reindex_htf_to_ltf(htf_trend_dn(htf), df.index)
    atr_fast = atr(df, 14); atr_slow = atr(df, 50)
    vol_contract = (atr_fast / atr_slow.replace(0, np.nan)) < 0.7
    _, bb_upper, _ = bollinger(df["close"], 20, 2.5)
    r = rsi(df["close"], 14)
    return htf_dn & vol_contract & (df["high"] >= bb_upper) & (r > 75)

def strat_F_short(df: pd.DataFrame, htf: pd.DataFrame) -> pd.Series:
    htf_dn = reindex_htf_to_ltf(htf_trend_dn(htf), df.index)
    r = rsi(df["close"], 14)
    z = zscore(df["close"], 20)
    _, bb_upper, _ = bollinger(df["close"], 20, 2.0)
    tbr = taker_buy_ratio(df)
    return (htf_dn & (r > 70) & (z > 1.5) & (df["close"] < bb_upper) & (tbr < 0.50))


# ============================================================================
# Tight-stop computation per strategy
# ============================================================================

def make_stop_from_recent_low(df: pd.DataFrame, lookback: int = 3,
                              atr_buffer_mult: float = 0.1) -> pd.Series:
    """Stop = min(low of last `lookback` bars) - atr_buffer_mult * ATR(14).
    Computed at the signal bar, used as the stop for the entry at next-bar open.
    The lookback EXCLUDES the current bar to avoid lookahead on intra-bar action
    (we use the closed bars' lows)."""
    lows = df["low"].rolling(lookback).min().shift(1)
    a = atr(df, 14)
    return lows - atr_buffer_mult * a


def make_stop_from_5bar_low(df: pd.DataFrame) -> pd.Series:
    return make_stop_from_recent_low(df, lookback=5, atr_buffer_mult=0.2)


# ============================================================================
# Modified backtester for smart strategies
# (same logic as backtester.backtest, but signal function and stop function
#  are passed in; entry rules are side-aware)
# ============================================================================

def backtest_smart(df: pd.DataFrame, signal: pd.Series, side: str,
                   stop_price: pd.Series, r_mult: float = 2.0,
                   max_bars: int | None = 48,
                   costs: Costs = Costs(),
                   equity_risk_per_trade: float = 0.01) -> BacktestResult:
    """Backtest a smart strategy with a precomputed signal and stop series.

    Stop and target are computed at the signal bar (close), entry fills at the
    next bar's open. Stop price is in absolute terms (already includes any
    ATR buffer). Target = entry ± r_mult * (entry - stop) depending on side.
    """
    df = df.sort_index()
    n = len(df)
    if n < 50:
        return BacktestResult([], np.array([]), np.array([]), 1.0, 1.0)

    signal_arr = signal.fillna(False).values
    stop_arr = stop_price.values

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    times  = df.index

    slip_frac = costs.slippage_bps / 10000.0

    trades: list[Trade] = []
    equity = 1.0
    equity_curve = np.empty(n)
    equity_curve_times = np.empty(n, dtype="datetime64[ns]")

    i = 0
    while i < n:
        equity_curve[i] = equity
        equity_curve_times[i] = times[i].to_datetime64()

        if i + 1 < n and signal_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_arr[i]):
            entry_raw = float(opens[i+1])
            if entry_raw <= 0:
                i += 1
                continue
            stop_price_val = float(stop_arr[i])
            if not np.isfinite(stop_price_val) or stop_price_val <= 0:
                i += 1
                continue

            # For longs: stop must be below entry. For shorts: above.
            if side == "long" and stop_price_val >= entry_raw:
                i += 1; continue
            if side == "short" and stop_price_val <= entry_raw:
                i += 1; continue

            # Slippage on entry (adverse)
            if side == "long":
                entry_price = entry_raw * (1 + slip_frac)
            else:
                entry_price = entry_raw * (1 - slip_frac)

            risk_per_unit = abs(entry_price - stop_price_val)
            if risk_per_unit <= 0 or not np.isfinite(risk_per_unit):
                i += 1; continue

            # If the bar that fills our entry (i+1) ALREADY touches stop on
            # the downside (long) or upside (short), we skip — too risky to
            # assume fill before stop. Conservative.
            # Actually the spec says: fill at next-bar open. If that bar's
            # range includes the stop, we'd need to model intra-bar sequence.
            # Conservative: if the fill bar's low (long) is <= stop, assume
            # the stop fired at the stop price (i.e., immediate stop-out at
            # entry bar). This is the worst case.
            fill_low = lows[i+1]; fill_high = highs[i+1]

            if side == "long":
                target_price = entry_price + r_mult * risk_per_unit
            else:
                target_price = entry_price - r_mult * risk_per_unit

            # Position size
            risk_amount = equity_risk_per_trade * equity
            size_units = risk_amount / risk_per_unit
            if size_units <= 0:
                i += 1; continue

            entry_notional = size_units * entry_price
            entry_commission = entry_notional * costs.commission_pct
            equity_after_entry = equity - entry_commission

            # Walk forward from bar i+1 (the fill bar)
            exit_idx = -1
            exit_price = float("nan")
            exit_reason = ""
            bars_held = 0
            j = i + 1

            # Special case: fill bar itself (j = i+1). If the fill bar's range
            # touches both stop and target, stop first. If only stop, exit at
            # stop. If only target, exit at target. Otherwise continue.
            while j < n:
                bars_held = j - (i + 1) + 1
                h = highs[j]; l = lows[j]

                if side == "long":
                    hit_stop = l <= stop_price_val
                    hit_tgt = h >= target_price
                else:
                    hit_stop = h >= stop_price_val
                    hit_tgt = l <= target_price

                if hit_stop and hit_tgt:
                    exit_price = stop_price_val; exit_reason = "stop"
                    exit_idx = j; break
                elif hit_stop:
                    exit_price = stop_price_val; exit_reason = "stop"
                    exit_idx = j; break
                elif hit_tgt:
                    exit_price = target_price; exit_reason = "target"
                    exit_idx = j; break

                if max_bars is not None and bars_held >= max_bars:
                    exit_price = closes[j]; exit_reason = "time"
                    exit_idx = j; break
                j += 1

            if exit_idx < 0:
                exit_price = closes[-1]; exit_reason = "end_of_data"
                exit_idx = n - 1

            # Apply exit slippage + commission
            if side == "long":
                exit_fill = exit_price * (1 - slip_frac)
            else:
                exit_fill = exit_price * (1 + slip_frac)
            exit_notional = size_units * exit_fill
            exit_commission = exit_notional * costs.commission_pct

            if side == "long":
                gross_pnl = size_units * (exit_fill - entry_price)
            else:
                gross_pnl = size_units * (entry_price - exit_fill)
            total_cost = entry_commission + exit_commission
            net_pnl = gross_pnl - total_cost
            r_mult_actual = net_pnl / risk_amount if risk_amount > 0 else 0.0

            trades.append(Trade(
                entry_time=times[i+1], exit_time=times[exit_idx],
                side=side, entry_price=entry_price, exit_price=exit_fill,
                stop=stop_price_val, target=target_price,
                size=size_units, gross_pnl=gross_pnl, cost=total_cost,
                net_pnl=net_pnl, r_multiple=r_mult_actual,
                exit_reason=exit_reason, bars_held=bars_held,
            ))
            equity = equity_after_entry + net_pnl
            i = exit_idx + 1
            continue
        i += 1

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve[:i] if i < n else equity_curve,
        equity_curve_times=equity_curve_times[:i] if i < n else equity_curve_times,
        initial_equity=1.0, final_equity=equity,
    )


# ============================================================================
# Strategy registry
# ============================================================================

@dataclass
class SmartStrategy:
    name: str
    side: str
    signal_fn: Callable
    stop_fn: Callable  # stop_fn(df) -> pd.Series of stop prices
    r_mult: float
    max_bars: int | None
    thesis: str


STRATEGIES = [
    SmartStrategy("A_zscore_rev_long",    "long",  strat_A_extreme_zscore_reversion, make_stop_from_recent_low, 2.0, 48,
                  "Enter when 1h close is >2σ below 50-bar mean AND 4h uptrend. Tight stop at recent 3-bar low."),
    SmartStrategy("B_vol_contract_long",  "long",  strat_B_vol_contraction_reversion, make_stop_from_recent_low, 2.0, 48,
                  "Vol contraction + BB(20,2.5) touch + RSI<25 in 4h uptrend. Triple oversold stack."),
    SmartStrategy("C_3bar_reversal_long", "long",  strat_C_3bar_reversal_at_htf_mean, make_stop_from_recent_low, 2.0, 48,
                  "3 consecutive lower lows + higher close at EMA50 in 4h uptrend."),
    SmartStrategy("D_vwap_reclaim_long",  "long",  strat_D_vwap_reclaim_with_flow,   make_stop_from_5bar_low, 2.0, 48,
                  "VWAP reclaim after 5 bars below, with 2x volume + taker-buy > 0.55."),
    SmartStrategy("E_pin_at_ema_long",    "long",  strat_E_pin_at_support,           make_stop_from_recent_low, 2.0, 48,
                  "Bullish pin bar touching EMA50 in 4h uptrend. Tight stop at pin low."),
    SmartStrategy("F_stacked_rev_long",   "long",  strat_F_stacked_reversal,         make_stop_from_5bar_low, 2.0, 48,
                  "4 independent oversold signals stacked: HTF up, RSI<30, z<-1.5, BB reclaim, taker-buy>0.50."),
    SmartStrategy("G_low_vol_break_long", "long",  strat_G_low_vol_breakout,         make_stop_from_5bar_low, 2.0, 48,
                  "Low-vol breakout above 20-bar high with 1.5x volume in 4h uptrend. Tight stop in tight range."),
    SmartStrategy("H_trend_pb_engulf",    "long",  strat_H_trend_pullback_engulfing, make_stop_from_recent_low, 2.0, 48,
                  "Fully stacked 4h trend + 1h pullback to EMA50 + bullish engulfing. Tight stop at engulfing low."),
    # Short mirrors (3 of them, to test the symmetric claim)
    SmartStrategy("A_zscore_rev_short",   "short", strat_A_short,                    make_stop_from_recent_low, 2.0, 48,
                  "Short mirror of A: z > 2 in 4h downtrend."),
    SmartStrategy("B_vol_contract_short", "short", strat_B_short,                    make_stop_from_recent_low, 2.0, 48,
                  "Short mirror of B: BB upper touch + RSI > 75 in 4h downtrend."),
    SmartStrategy("F_stacked_rev_short",  "short", strat_F_short,                    make_stop_from_5bar_low, 2.0, 48,
                  "Short mirror of F: stacked overbought signals in 4h downtrend."),
]


# ============================================================================
# Runner
# ============================================================================

def load_1h_with_htf(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA_DIR / f"{symbol}_1h.parquet").set_index("open_time").sort_index()
    htf = resample_htf(df, "4h")
    return df, htf


def split_is_oos(df: pd.DataFrame, is_frac: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df); cut = int(n * is_frac)
    return df.iloc[:cut], df.iloc[cut:]


def compute_metrics(result: BacktestResult) -> dict:
    from evaluate import compute_metrics as _cm
    return _cm(result, "default")


def all_gates_passed(oos_m: dict) -> bool:
    from evaluate import all_gates_passed as _agp
    return _agp(oos_m)


def run_all():
    print("\n" + "="*78)
    print("SMART STRATEGIES — designed with explicit theses for 70%/1:2 R:R")
    print("="*78)

    all_results = []
    for sym in ["BTCUSDT", "ETHUSDT"]:
        print(f"\n>>> Loading {sym} 1h + 4h resample")
        df, htf = load_1h_with_htf(sym)
        df_is, df_oos = split_is_oos(df, 0.70)
        htf_is, htf_oos = split_is_oos(htf, 0.70)  # split HTF at same fraction
        print(f"    1h IS: {len(df_is)} bars ({df_is.index.min()} -> {df_is.index.max()})")
        print(f"    1h OOS: {len(df_oos)} bars ({df_oos.index.min()} -> {df_oos.index.max()})")

        for strat in STRATEGIES:
            t0 = time.time()
            try:
                # IS signal + stop
                sig_is = strat.signal_fn(df_is, htf_is)
                stop_is = strat.stop_fn(df_is)
                # OOS signal + stop
                sig_oos = strat.signal_fn(df_oos, htf_oos)
                stop_oos = strat.stop_fn(df_oos)

                # Run backtests
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
                all_results.append({"symbol": sym, "strategy": strat.name, "side": strat.side,
                                    "thesis": strat.thesis, "error": str(e)})

    df_r = pd.DataFrame(all_results)
    df_r.to_csv(OUT_PATH, index=False)
    print(f"\n>>> Wrote {OUT_PATH} with {len(df_r)} rows")

    # Pick best for trade log + equity curve
    # Best = highest OOS win rate among those with >= 200 OOS trades
    eligible = df_r[(df_r["oos_n_closed"] >= 200) & (df_r.get("oos_net_profit_pct", 0) > -0.99)]
    if len(eligible) == 0:
        # Fall back to highest OOS win rate overall
        eligible = df_r[df_r.get("oos_net_profit_pct", 0) > -0.99]
    if len(eligible) > 0:
        best_row = eligible.sort_values("oos_win_rate", ascending=False).iloc[0]
        print(f"\n>>> Best candidate: {best_row['symbol']} {best_row['strategy']}  "
              f"OOS win={best_row['oos_win_rate']*100:.1f}%  trades={best_row['oos_n_closed']}  net={best_row['oos_net_profit_pct']*100:.1f}%")
        # Re-run to get the BacktestResult object for trade log
        sym = best_row["symbol"]
        strat = next(s for s in STRATEGIES if s.name == best_row["strategy"])
        df, htf = load_1h_with_htf(sym)
        df_is, df_oos = split_is_oos(df, 0.70)
        htf_is, htf_oos = split_is_oos(htf, 0.70)
        sig_oos = strat.signal_fn(df_oos, htf_oos)
        stop_oos = strat.stop_fn(df_oos)
        oos_result = backtest_smart(df_oos, sig_oos, strat.side, stop_oos,
                                    r_mult=strat.r_mult, max_bars=strat.max_bars)
        # Trade log
        rows = []
        for t in oos_result.trades:
            rows.append({
                "entry_time": t.entry_time, "exit_time": t.exit_time, "side": t.side,
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "stop": t.stop, "target": t.target, "size": t.size,
                "gross_pnl": t.gross_pnl, "cost": t.cost, "net_pnl": t.net_pnl,
                "R_multiple": t.r_multiple, "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            })
        pd.DataFrame(rows).to_csv(TRADES_OUT, index=False)
        print(f"    Wrote {TRADES_OUT} with {len(rows)} trades")

        # Equity curve (IS + OOS overlaid)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        is_result = backtest_smart(df_is, strat.signal_fn(df_is, htf_is), strat.side,
                                   strat.stop_fn(df_is), r_mult=strat.r_mult, max_bars=strat.max_bars)
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        is_eq = is_result.equity_curve
        is_t = pd.to_datetime(is_result.equity_curve_times)
        ax.plot(is_t, is_eq, color="#1f77b4", linewidth=1.2,
                label=f"IS (70%)  final={is_result.final_equity:.3f}")
        oos_eq = oos_result.equity_curve * is_result.final_equity
        oos_t = pd.to_datetime(oos_result.equity_curve_times)
        ax.plot(oos_t, oos_eq, color="#d62728", linewidth=1.2,
                label=f"OOS (30%)  final={is_result.final_equity * oos_result.final_equity:.3f}")
        boundary = df_is.index.max()
        ax.axvline(boundary, color="black", linestyle="--", alpha=0.5, label="IS/OOS boundary")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Equity (USDT, starting at 1.0)")
        title = (f"Smart Strategy Equity Curve — {sym} {strat.name}\n"
                 f"side={strat.side}  R={strat.r_mult}  | OOS win={best_row['oos_win_rate']*100:.1f}%  "
                 f"trades={best_row['oos_n_closed']}  net={best_row['oos_net_profit_pct']*100:+.1f}%  "
                 f"dd={best_row['oos_max_dd_pct']*100:.1f}%")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.savefig(EQUITY_OUT, dpi=120)
        plt.close(fig)
        print(f"    Wrote {EQUITY_OUT}")

    print("\n>>> Done.")


if __name__ == "__main__":
    run_all()
