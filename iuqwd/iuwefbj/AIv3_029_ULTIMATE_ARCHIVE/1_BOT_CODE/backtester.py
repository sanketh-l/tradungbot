"""
Phase 3: Event-driven backtester.

Design principles (correctness first):
  - All indicators are precomputed using pandas rolling/ewm operations that
    by construction only look at bars up to and including the current bar
    (closed='right' semantics). No future data is accessed.
  - Signals computed at bar i (using close of bar i) → entries fill at bar
    i+1 open (with slippage + commission). This eliminates signal-bar peeking.
  - Stop/target checks happen from the entry bar (i+1) onward, using
    intra-bar high/low. If a single bar hits both stop and target, the stop
    is assumed first (conservative).
  - One position at a time. No pyramiding.
  - Costs: commission_pct (per side, default 0.05% = 5 bps) + slippage_bps
    (per side, default 1 bp). Both applied on entry and exit.
  - Position sizing: 1% equity risk per trade (risk = |entry - stop| * size).
  - Equity is tracked in quote currency (USDT), starting at 1.0 (unit).

Win/loss definitions (per spec, strict):
  - Win  = exit_reason == "target" AND net_pnl > 0
  - Loss = exit_reason == "stop"   AND net_pnl < 0
  - Other exits (trail, time, end_of_data) are closed trades but neither win
    nor loss for win-rate purposes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Costs:
    commission_pct: float = 0.0005  # 0.05% per side
    slippage_bps:   float = 1.0     # 1 basis point per side (≈1 tick on BTC/ETH)

    def doubled(self) -> "Costs":
        return Costs(
            commission_pct=self.commission_pct * 2,
            slippage_bps=self.slippage_bps * 2,
        )


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time:  pd.Timestamp
    side: str               # "long" / "short"
    entry_price: float      # actual fill (after slippage)
    exit_price:  float      # actual fill (after slippage)
    stop: float             # stop price (gross)
    target: float           # target price (gross)
    size: float             # position size in base units
    gross_pnl: float        # before costs
    cost: float             # total commission + slippage cost
    net_pnl: float          # gross_pnl - cost
    r_multiple: float       # net_pnl / (initial_risk_per_unit * size)
    exit_reason: str        # "target", "stop", "trail", "time", "end_of_data"
    bars_held: int


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: np.ndarray   # equity at each bar (only meaningful at trade-boundary bars)
    equity_curve_times: np.ndarray
    initial_equity: float
    final_equity: float

    def n_closed(self) -> int:
        return len(self.trades)

    def n_wins(self) -> int:
        return sum(1 for t in self.trades if t.exit_reason == "target" and t.net_pnl > 0)

    def n_losses(self) -> int:
        return sum(1 for t in self.trades if t.exit_reason == "stop" and t.net_pnl < 0)


# ============================================================================
# Indicator computation (no lookahead by construction)
# ============================================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]; low = df["low"]; close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Standard ADX (Wilder). Returns just the ADX line."""
    high = df["high"]; low = df["low"]; close = df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm  = pd.Series(plus_dm,  index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx_line


def donchian(df: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series]:
    """Returns (upper, lower) — both EXCLUDE the current bar to avoid lookahead."""
    upper = df["high"].rolling(length).max().shift(1)
    lower = df["low"].rolling(length).min().shift(1)
    return upper, lower


def rolling_linreg_slope(series: pd.Series, length: int) -> pd.Series:
    """Slope of best-fit line over `length` bars, ending at and including current bar."""
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    # rolling sum of x*y and y
    y = series.rolling(length)
    y_mean = y.mean()
    xy_sum = (series.rolling(length).apply(lambda w: ((x * w).sum()), raw=True))
    num = xy_sum - length * x_mean * y_mean
    slope = num / x_var
    return slope


def vwap_rolling(df: pd.DataFrame, length: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return (typical * df["volume"]).rolling(length).sum() / df["volume"].rolling(length).sum().replace(0, np.nan)


def bollinger(series: pd.Series, length: int, n_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(length).mean()
    sd  = series.rolling(length).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return mid, upper, lower


# ============================================================================
# Signal generation (vectorized, no lookahead)
# ============================================================================

def compute_indicators(df: pd.DataFrame, cfg: dict) -> dict:
    """Precompute all indicators needed by the config. Each indicator value at
    index i uses only data from bars [..., i]."""
    ind: dict[str, Any] = {}
    close = df["close"]

    # Trend filter precomputation
    tf = cfg["trend_filter"]
    if tf["kind"] == "ema_cross":
        ind["ema_fast"] = ema(close, tf["fast"])
        ind["ema_slow"] = ema(close, tf["slow"])
    elif tf["kind"] == "donchian":
        ind["donch_upper"], ind["donch_lower"] = donchian(df, tf["length"])
    elif tf["kind"] == "adx":
        ind["adx"] = adx(df, tf["period"])
        # also need +DI/-DI for direction
        high = df["high"]; low = df["low"]
        up = high.diff(); down = -low.diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        plus_dm  = pd.Series(plus_dm,  index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr_ = tr.ewm(alpha=1/tf["period"], adjust=False, min_periods=tf["period"]).mean()
        ind["plus_di"]  = 100 * plus_dm.ewm(alpha=1/tf["period"], adjust=False, min_periods=tf["period"]).mean() / atr_.replace(0, np.nan)
        ind["minus_di"] = 100 * minus_dm.ewm(alpha=1/tf["period"], adjust=False, min_periods=tf["period"]).mean() / atr_.replace(0, np.nan)
    elif tf["kind"] == "htf_ema":
        ind["htf_ema"] = ema(close, tf["htf_len"])
        ind["htf_slope"] = ind["htf_ema"].diff(tf["slope_window"])
    elif tf["kind"] == "linreg":
        ind["linreg_slope"] = rolling_linreg_slope(close, tf["length"])

    # Entry trigger precomputation
    et = cfg["entry_trigger"]
    if et["kind"] == "pullback_ema":
        ind["pullback_ema"] = ema(close, et["ema_len"])
    elif et["kind"] == "breakout_nbar":
        ind["breakout_high"] = df["high"].rolling(et["length"]).max().shift(1)
        ind["breakout_low"]  = df["low"].rolling(et["length"]).min().shift(1)
    elif et["kind"] == "rsi_reversal":
        ind["rsi_entry"] = rsi(close, et["rsi_len"])
        ind["rsi_entry_prev"] = ind["rsi_entry"].shift(1)
    elif et["kind"] == "bb_touch":
        ind["bb_mid"], ind["bb_upper"], ind["bb_lower"] = bollinger(close, et["bb_len"], et["n_std"])
    elif et["kind"] == "vwap_reclaim":
        ind["vwap"] = vwap_rolling(df, et["vwap_len"])
        ind["vwap_prev"] = ind["vwap"].shift(1)
        ind["close_prev"] = close.shift(1)

    # Confirmation precomputation
    cf = cfg["confirmation"]
    if cf["kind"] == "volume_avg":
        ind["vol_ma"] = df["volume"].rolling(cf["vol_len"]).mean()
    elif cf["kind"] == "rsi_band":
        ind["rsi_conf"] = rsi(close, cf["rsi_len"])
    elif cf["kind"] == "atr_expand":
        ind["atr_conf"] = atr(df, cf["atr_len"])
        ind["atr_ma"]   = ind["atr_conf"].rolling(cf["atr_len"]).mean()

    # Stop precomputation
    st = cfg["stop"]
    if st["kind"] == "atr_mult":
        ind["atr_stop"] = atr(df, st["atr_len"])
    elif st["kind"] == "swing":
        ind["swing_low"]  = df["low"].rolling(st["lookback"]).min().shift(1)
        ind["swing_high"] = df["high"].rolling(st["lookback"]).max().shift(1)

    return ind


def trend_filter_signal(df: pd.DataFrame, cfg: dict, ind: dict) -> pd.Series:
    """Returns +1 for uptrend, -1 for downtrend, 0 for none.
    Value at index i is computed using only data up to and including bar i."""
    tf = cfg["trend_filter"]
    close = df["close"]
    sig = pd.Series(0, index=df.index, dtype=int)

    if tf["kind"] == "ema_cross":
        up = (ind["ema_fast"] > ind["ema_slow"]) & (close > ind["ema_slow"])
        dn = (ind["ema_fast"] < ind["ema_slow"]) & (close < ind["ema_slow"])
        sig[up] = 1; sig[dn] = -1
    elif tf["kind"] == "donchian":
        up = close > ind["donch_upper"]
        dn = close < ind["donch_lower"]
        sig[up] = 1; sig[dn] = -1
    elif tf["kind"] == "adx":
        adx_line = ind["adx"]
        strong = adx_line > tf["threshold"]
        up = strong & (ind["plus_di"] > ind["minus_di"])
        dn = strong & (ind["minus_di"] > ind["plus_di"])
        sig[up] = 1; sig[dn] = -1
    elif tf["kind"] == "htf_ema":
        up = ind["htf_slope"] > 0
        dn = ind["htf_slope"] < 0
        sig[up] = 1; sig[dn] = -1
    elif tf["kind"] == "linreg":
        up = ind["linreg_slope"] > 0
        dn = ind["linreg_slope"] < 0
        sig[up] = 1; sig[dn] = -1
    return sig


def entry_trigger_signal(df: pd.DataFrame, cfg: dict, ind: dict, side: str) -> pd.Series:
    """Returns bool series: True if entry trigger fires at this bar's close.
    The signal is side-agnostic in shape but the caller already restricts to
    one side via config; here we just compute the pattern for that side."""
    et = cfg["entry_trigger"]
    close = df["close"]; high = df["high"]; low = df["low"]; op = df["open"]
    sig = pd.Series(False, index=df.index)

    if et["kind"] == "pullback_ema":
        ema_line = ind["pullback_ema"]
        dist = (close - ema_line).abs() / ema_line
        if side == "long":
            sig = (low <= ema_line) & (close > ema_line) & (dist * 100 <= et["max_dist_pct"])
        else:
            sig = (high >= ema_line) & (close < ema_line) & (dist * 100 <= et["max_dist_pct"])
    elif et["kind"] == "breakout_nbar":
        if side == "long":
            sig = close > ind["breakout_high"]
        else:
            sig = close < ind["breakout_low"]
    elif et["kind"] == "rsi_reversal":
        r = ind["rsi_entry"]; rp = ind["rsi_entry_prev"]
        if side == "long":
            sig = (rp < et["oversold"]) & (r > et["oversold"]) & (r < et["overbought"])
        else:
            sig = (rp > (100 - et["oversold"])) & (r < (100 - et["oversold"])) & (r > (100 - et["overbought"]))
    elif et["kind"] == "bb_touch":
        if side == "long":
            sig = low <= ind["bb_lower"]
        else:
            sig = high >= ind["bb_upper"]
    elif et["kind"] == "vwap_reclaim":
        v = ind["vwap"]; vp = ind["vwap_prev"]; cp = ind["close_prev"]
        if side == "long":
            sig = (cp < vp) & (close > v)
        else:
            sig = (cp > vp) & (close < v)
    elif et["kind"] == "engulfing":
        body = (close - op)
        body_prev = close.shift(1) - op.shift(1)
        if side == "long":
            sig = (body_prev < 0) & (body > 0) & (close > op.shift(1)) & (op < close.shift(1))
        else:
            sig = (body_prev > 0) & (body < 0) & (close < op.shift(1)) & (op > close.shift(1))
    elif et["kind"] == "pin_bar":
        body = (close - op).abs()
        rng = (high - low).replace(0, np.nan)
        upper_wick = high - np.maximum(close, op)
        lower_wick = np.minimum(close, op) - low
        if side == "long":
            sig = (lower_wick > 2 * body) & (lower_wick > 0.6 * rng) & (close > op)
        else:
            sig = (upper_wick > 2 * body) & (upper_wick > 0.6 * rng) & (close < op)
    return sig


def confirmation_signal(df: pd.DataFrame, cfg: dict, ind: dict) -> pd.Series:
    """Returns bool series: True if confirmation passes."""
    cf = cfg["confirmation"]
    if cf["kind"] == "none":
        return pd.Series(True, index=df.index)
    if cf["kind"] == "volume_avg":
        return df["volume"] > cf["mult"] * ind["vol_ma"]
    if cf["kind"] == "rsi_band":
        return (ind["rsi_conf"] > cf["lo"]) & (ind["rsi_conf"] < cf["hi"])
    if cf["kind"] == "atr_expand":
        return ind["atr_conf"] > cf["mult"] * ind["atr_ma"]
    if cf["kind"] == "session":
        h = pd.Series(df.index.hour, index=df.index)  # tz-aware UTC index
        if cf["start_h"] < cf["end_h"]:
            return (h >= cf["start_h"]) & (h < cf["end_h"])
        else:
            return (h >= cf["start_h"]) | (h < cf["end_h"])
    return pd.Series(True, index=df.index)


# ============================================================================
# Event-driven backtest loop
# ============================================================================

def compute_stop_price(cfg: dict, ind: dict, i: int, entry_price: float,
                       side: str, df: pd.DataFrame) -> float:
    """Compute stop price (gross, before slippage on the exit side)."""
    st = cfg["stop"]
    if st["kind"] == "atr_mult":
        atr_val = ind["atr_stop"].iloc[i]
        if not np.isfinite(atr_val) or atr_val <= 0:
            atr_val = entry_price * 0.01  # fallback
        if side == "long":
            return entry_price - st["mult"] * atr_val
        else:
            return entry_price + st["mult"] * atr_val
    elif st["kind"] == "swing":
        if side == "long":
            sw = ind["swing_low"].iloc[i]
            return float(sw) if np.isfinite(sw) else entry_price * 0.99
        else:
            sw = ind["swing_high"].iloc[i]
            return float(sw) if np.isfinite(sw) else entry_price * 1.01
    elif st["kind"] == "fixed_pct":
        if side == "long":
            return entry_price * (1 - st["pct"])
        else:
            return entry_price * (1 + st["pct"])
    raise ValueError(st["kind"])


def backtest(df: pd.DataFrame, cfg: dict, costs: Costs,
             equity_risk_per_trade: float = 0.01) -> BacktestResult:
    """Run a full event-driven backtest.

    df: full OHLCV slice (already filtered to the test window). Index must be
        tz-aware UTC DatetimeIndex.
    cfg: strategy configuration dict.
    costs: commission + slippage spec.
    """
    df = df.sort_index()
    n = len(df)
    if n < 50:
        return BacktestResult([], np.array([]), np.array([]), 1.0, 1.0)

    side = cfg["side"]
    r_mult = cfg["target"]["r_mult"]
    exit_cfg = cfg["exit"]

    # Precompute indicators (vectorized, no lookahead).
    ind = compute_indicators(df, cfg)
    trend_sig = trend_filter_signal(df, cfg, ind).values
    entry_sig = entry_trigger_signal(df, cfg, ind, side).values
    conf_sig  = confirmation_signal(df, cfg, ind).values

    # Combined signal at bar i: triggers only if all three agree AND trend
    # filter points in the same direction as the side.
    want = 1 if side == "long" else -1
    signal = (entry_sig & conf_sig & (trend_sig == want))

    # Bar arrays (numpy for speed)
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    times  = df.index

    # Slippage as a fraction
    slip_frac = costs.slippage_bps / 10000.0

    trades: list[Trade] = []
    equity = 1.0
    peak_equity = 1.0
    equity_curve = np.empty(n)
    equity_curve_times = np.empty(n, dtype="datetime64[ns]")

    i = 0
    while i < n:
        # Mark current equity (held flat between trades)
        equity_curve[i] = equity
        equity_curve_times[i] = times[i].to_datetime64()

        # Check if a signal fired at bar i (close). If so, fill at bar i+1 open.
        if i + 1 < n and signal[i] and np.isfinite(opens[i+1]):
            entry_raw = float(opens[i+1])
            if not np.isfinite(entry_raw) or entry_raw <= 0:
                i += 1
                continue
            # Apply slippage on entry (adverse direction)
            if side == "long":
                entry_price = entry_raw * (1 + slip_frac)
            else:
                entry_price = entry_raw * (1 - slip_frac)

            # Compute stop and target from the actual fill price.
            stop_price = compute_stop_price(cfg, ind, i, entry_price, side, df)
            risk_per_unit = abs(entry_price - stop_price)
            if risk_per_unit <= 0 or not np.isfinite(risk_per_unit):
                i += 1
                continue
            if side == "long":
                target_price = entry_price + r_mult * risk_per_unit
            else:
                target_price = entry_price - r_mult * risk_per_unit

            # Position size: 1% equity risk
            risk_amount = equity_risk_per_trade * equity
            size_units = risk_amount / risk_per_unit
            if size_units <= 0 or not np.isfinite(size_units):
                i += 1
                continue

            # Commission on entry (on notional)
            entry_notional = size_units * entry_price
            entry_commission = entry_notional * costs.commission_pct
            equity_after_entry = equity - entry_commission
            # Slippage on entry already priced into entry_price.

            # Walk forward from bar i+1 to find exit
            exit_idx = -1
            exit_price = float("nan")
            exit_reason = ""
            bars_held = 0

            # Trailing stop state
            trail_active = False
            trail_stop = stop_price  # updated if trailing activates

            # Time exit
            max_bars = exit_cfg.get("max_bars", None)
            trail_kind = exit_cfg["kind"]

            j = i + 1
            while j < n:
                bars_held = j - (i + 1) + 1
                h = highs[j]; l = lows[j]; c = closes[j]

                # Determine effective stop for this bar (could be trailing)
                eff_stop = trail_stop if trail_active else stop_price

                # Conservative: check stop first when both hit in same bar.
                hit_stop = False
                hit_target = False
                if side == "long":
                    if l <= eff_stop:
                        hit_stop = True
                    if h >= target_price:
                        hit_target = True
                else:
                    if h >= eff_stop:
                        hit_stop = True
                    if l <= target_price:
                        hit_target = True

                if hit_stop and hit_target:
                    # Stop first (conservative)
                    exit_price = eff_stop
                    exit_reason = "stop"
                    exit_idx = j
                    break
                elif hit_stop:
                    exit_price = eff_stop
                    exit_reason = "stop" if not trail_active else "trail"
                    exit_idx = j
                    break
                elif hit_target:
                    exit_price = target_price
                    exit_reason = "target"
                    exit_idx = j
                    break

                # Trailing logic: activate after 1R move in favor, then trail.
                if trail_kind == "trail_after_1R":
                    if side == "long":
                        if h >= entry_price + risk_per_unit:
                            trail_active = True
                            # trail at 0.5R below the best price seen so far
                            new_stop = h - exit_cfg.get("trail_r", 0.5) * risk_per_unit
                            if new_stop > trail_stop:
                                trail_stop = new_stop
                    else:
                        if l <= entry_price - risk_per_unit:
                            trail_active = True
                            new_stop = l + exit_cfg.get("trail_r", 0.5) * risk_per_unit
                            if new_stop < trail_stop:
                                trail_stop = new_stop

                # Time exit
                if max_bars is not None and bars_held >= max_bars:
                    exit_price = c
                    exit_reason = "time"
                    exit_idx = j
                    break

                j += 1

            if exit_idx < 0:
                # End of data: close at last close
                exit_price = closes[-1]
                exit_reason = "end_of_data"
                exit_idx = n - 1

            # Apply exit slippage (adverse) and commission
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
            total_cost = entry_commission + exit_commission + size_units * (slip_frac * entry_price + slip_frac * exit_fill)
            net_pnl = gross_pnl - total_cost
            r_mult_actual = net_pnl / risk_amount if risk_amount > 0 else 0.0

            trades.append(Trade(
                entry_time=times[i+1],
                exit_time=times[exit_idx],
                side=side,
                entry_price=entry_price,
                exit_price=exit_fill,
                stop=stop_price,
                target=target_price,
                size=size_units,
                gross_pnl=gross_pnl,
                cost=total_cost,
                net_pnl=net_pnl,
                r_multiple=r_mult_actual,
                exit_reason=exit_reason,
                bars_held=bars_held,
            ))
            equity = equity_after_entry + net_pnl
            peak_equity = max(peak_equity, equity)

            # Skip ahead to bar after exit (no overlapping trades)
            i = exit_idx + 1
            continue

        i += 1

    final_equity = equity
    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve[:i] if i < n else equity_curve,
        equity_curve_times=equity_curve_times[:i] if i < n else equity_curve_times,
        initial_equity=1.0,
        final_equity=final_equity,
    )


# ============================================================================
# Fast Pass 1 filter (vectorized, conservative-on-the-cheap-side approximation)
# ============================================================================

def pass1_fast_filter(df: pd.DataFrame, cfg: dict, min_trades: int = 30) -> tuple[bool, dict]:
    """Quick vectorized signal-frequency + gross-edge check on a 6-month slice.

    Returns (passes, stats).
    'passes' is True if the strategy generates >= min_trades signals AND has
    positive gross profit (optimistic estimate — no costs, no conservative
    stop-first rule, bar-high/low based). This is intentionally optimistic so
    we don't reject borderline-but-real strategies; the full backtest will
    give the strict verdict.
    """
    n = len(df)
    if n < 50:
        return False, {"reason": "too_short", "n_signals": 0, "gross_R": 0.0}

    side = cfg["side"]
    r_mult = cfg["target"]["r_mult"]

    try:
        ind = compute_indicators(df, cfg)
        trend_sig = trend_filter_signal(df, cfg, ind).values
        entry_sig = entry_trigger_signal(df, cfg, ind, side).values
        conf_sig  = confirmation_signal(df, cfg, ind).values
    except Exception:
        return False, {"reason": "indicator_error", "n_signals": 0, "gross_R": 0.0}

    want = 1 if side == "long" else -1
    signal = entry_sig & conf_sig & (trend_sig == want)
    n_signals = int(signal.sum())
    if n_signals < min_trades:
        return False, {"reason": "too_few_signals", "n_signals": n_signals, "gross_R": 0.0}

    # Fast gross-edge estimate: for each signal at bar i, look ahead up to 50
    # bars. Compute stop/target at bar i close (NOT i+1 open — slight optimism).
    # If high hits target before low hits stop, count +r_mult. If stop first,
    # count -1. If neither within 50 bars, skip (effectively a non-trade).
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    look = min(50, n - 1)
    gross_R = 0.0
    idxs = np.where(signal)[0]
    last_exit = -1
    n_eval = 0
    for i in idxs:
        if i + 1 >= n or i <= last_exit:
            continue
        # Use bar i close as entry proxy (fast version; full backtester uses i+1 open)
        entry = closes[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        # Stop from same bar
        stop_price = compute_stop_price(cfg, ind, i, entry, side, df)
        risk = abs(entry - stop_price)
        if risk <= 0 or not np.isfinite(risk):
            continue
        if side == "long":
            target_price = entry + r_mult * risk
        else:
            target_price = entry - r_mult * risk
        # Walk forward
        end_j = min(i + 1 + look, n)
        exit_r = 0.0
        for j in range(i+1, end_j):
            h = highs[j]; l = lows[j]
            if side == "long":
                hit_stop = l <= stop_price
                hit_tgt  = h >= target_price
            else:
                hit_stop = h >= stop_price
                hit_tgt  = l <= target_price
            if hit_stop and hit_tgt:
                exit_r = -1.0  # conservative
                break
            elif hit_stop:
                exit_r = -1.0
                break
            elif hit_tgt:
                exit_r = r_mult
                break
        if exit_r != 0.0:
            gross_R += exit_r
            n_eval += 1
            last_exit = j  # no overlap

    if n_eval < min_trades:
        return False, {"reason": "too_few_eval", "n_signals": n_signals, "n_eval": n_eval, "gross_R": gross_R}
    if gross_R <= 0:
        return False, {"reason": "negative_gross_R", "n_signals": n_signals, "n_eval": n_eval, "gross_R": gross_R}
    return True, {"reason": "ok", "n_signals": n_signals, "n_eval": n_eval, "gross_R": gross_R}


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    # Quick smoke test: load one config and a slice of data, run a backtest.
    cfgs = json.loads((DATA_DIR / "strategies.json").read_text())
    cfg = cfgs[0]
    print(f"Smoke test on config: {cfg['id']}  ({cfg['symbol']} {cfg['timeframe']} {cfg['side']})")

    df = pd.read_parquet(DATA_DIR / f"{cfg['symbol']}_{cfg['timeframe']}.parquet")
    df = df.set_index("open_time")
    # 6-month slice
    end = df.index.min() + pd.Timedelta(days=180)
    df_slice = df.loc[:end]
    print(f"  slice: {len(df_slice)} bars  {df_slice.index.min()} -> {df_slice.index.max()}")

    # Fast pass
    ok, stats = pass1_fast_filter(df_slice, cfg)
    print(f"  pass1: ok={ok}  stats={stats}")

    # Full backtest
    result = backtest(df_slice, cfg, Costs())
    print(f"  backtest: trades={result.n_closed()}  wins={result.n_wins()}  losses={result.n_losses()}")
    if result.trades:
        eq = result.final_equity
        print(f"  final equity: {eq:.4f}")
        # First 3 trades
        for t in result.trades[:3]:
            print(f"    {t.entry_time} -> {t.exit_time}  entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
                  f"R={t.r_multiple:.2f} reason={t.exit_reason}")
