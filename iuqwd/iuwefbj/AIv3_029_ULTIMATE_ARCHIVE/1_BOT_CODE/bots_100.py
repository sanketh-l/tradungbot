"""
100 Custom Bots — diverse strategy generator + 30-day test.

Generates 100 distinct trading strategies across:
  - Timeframes: 1h, 4h (swing-friendly)
  - Symbols: BTCUSDT, ETHUSDT
  - Sides: long, short, both
  - Entries: breakout, pullback, reversal, momentum, vol-squeeze, pin bar,
    engulfing, RSI reversal, BB touch, VWAP reclaim, 2-bar reversal
  - Stops: ATR mult (0.5-2.0), swing low/high, signal-bar low/high
  - Targets: 1.5R, 2.0R, 2.5R, 3.0R
  - Filters: session, volume surge, taker-buy, RSI band, ADX threshold, HTF trend

Then tests all 100 on last 30 days with $10,000 starting wallet, 1% risk/trade.
Reports top 10 by net profit with full trade logs in Excel.
"""
from __future__ import annotations

import itertools
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

from backtester import (
    Costs, BacktestResult, Trade,
    ema, rsi, atr, adx, bollinger, vwap_rolling, donchian,
)
from smart_strategies import (
    zscore, resample_htf, taker_buy_ratio, reindex_htf_to_ltf,
    htf_trend_up, htf_trend_dn,
)
from smart_strategies_round2 import (
    htf_strong_uptrend, htf_strong_downtrend,
    make_stop_signal_bar_low, make_stop_signal_bar_high,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOWNLOAD = Path(__file__).resolve().parent.parent / "download"

END_DATE = pd.Timestamp("2026-10-14 18:15:00", tz="UTC")
START_DATE = END_DATE - pd.Timedelta(days=30)

INITIAL_WALLET = 10000.0
RISK_PER_TRADE = 0.01
COSTS = Costs(commission_pct=0.0005, slippage_bps=1.0)


def load_window(symbol: str, timeframe: str):
    """Load data; if 4h requested, resample from 1h."""
    if timeframe == "4h":
        df_1h = pd.read_parquet(DATA_DIR / f"{symbol}_1h.parquet").set_index("open_time").sort_index()
        df = df_1h.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum", "quote_volume": "sum",
            "taker_buy_base": "sum",
        }).dropna()
    else:
        df = pd.read_parquet(DATA_DIR / f"{symbol}_{timeframe}.parquet").set_index("open_time").sort_index()
    warmup = START_DATE - pd.Timedelta(hours=400)
    df = df.loc[warmup:END_DATE].copy()
    # HTF is always 1h (regardless of LTF timeframe)
    df_1h_raw = pd.read_parquet(DATA_DIR / f"{symbol}_1h.parquet").set_index("open_time").sort_index()
    htf = df_1h_raw.loc[warmup:END_DATE].copy()
    return df, htf


def rolling_linreg_slope(series: pd.Series, length: int = 20) -> pd.Series:
    x = np.arange(length, dtype=float); xm = x.mean(); xv = ((x-xm)**2).sum()
    xy = series.rolling(length).apply(lambda w: (x*w).sum(), raw=True)
    return (xy - length*xm*series.rolling(length).mean()) / xv


# ============================================================================
# Strategy component library — used to assemble 100 distinct bots
# ============================================================================

def trend_filter(df, htf, kind):
    """Returns (trend_long, trend_short) boolean series."""
    close = df["close"]; high = df["high"]; low = df["low"]
    if kind == "ema_9_21":
        ef, es = ema(close, 9), ema(close, 21)
        return ef > es, ef < es
    if kind == "ema_20_50":
        e20, e50 = ema(close, 20), ema(close, 50)
        return e20 > e50, e20 < e50
    if kind == "ema_50_200":
        e50, e200 = ema(close, 50), ema(close, 200)
        return e50 > e200, e50 < e200
    if kind == "htf_ema_200":
        e200 = ema(close, 200)
        return close > e200, close < e200
    if kind == "htf_strong":
        htf_up = reindex_htf_to_ltf(htf_strong_uptrend(htf), df.index)
        htf_dn = reindex_htf_to_ltf(htf_strong_downtrend(htf), df.index)
        return htf_up, htf_dn
    if kind == "linreg_20":
        s = rolling_linreg_slope(close, 20)
        return s > 0, s < 0
    if kind == "adx_25":
        a = adx(df, 14)
        up = high.diff(); dn = -low.diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr_ = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_.replace(0, np.nan)
        return (a > 25) & (plus_di > minus_di), (a > 25) & (minus_di > plus_di)
    if kind == "none":
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
    raise ValueError(kind)


def entry_trigger(df, kind, side):
    """Returns boolean series — entry fires at bar close."""
    close = df["close"]; high = df["high"]; low = df["low"]; op = df["open"]
    rng = (high - low).replace(0, np.nan)
    body = close - op
    if kind == "breakout_10":
        u = high.rolling(10).max().shift(1); l = low.rolling(10).min().shift(1)
        return (close > u) if side == "long" else (close < l)
    if kind == "breakout_20":
        u = high.rolling(20).max().shift(1); l = low.rolling(20).min().shift(1)
        return (close > u) if side == "long" else (close < l)
    if kind == "breakout_50":
        u = high.rolling(50).max().shift(1); l = low.rolling(50).min().shift(1)
        return (close > u) if side == "long" else (close < l)
    if kind == "pullback_ema20":
        e = ema(close, 20)
        if side == "long":
            return (low <= e * 1.003) & (close > e) & (close > op)
        return (high >= e * 0.997) & (close < e) & (close < op)
    if kind == "pullback_ema50":
        e = ema(close, 50)
        if side == "long":
            return (low <= e * 1.004) & (close > e) & (close > op)
        return (high >= e * 0.996) & (close < e) & (close < op)
    if kind == "rsi_reversal":
        r = rsi(close, 14)
        if side == "long":
            return (r.shift(1) < 30) & (r > 30) & (r < 50)
        return (r.shift(1) > 70) & (r < 70) & (r > 50)
    if kind == "rsi2_extreme":
        r = rsi(close, 2)
        if side == "long":
            return r < 5
        return r > 95
    if kind == "bb_touch":
        _, u, l = bollinger(close, 20, 2.0)
        if side == "long":
            return low <= l
        return high >= u
    if kind == "vwap_reclaim":
        v = vwap_rolling(df, 50)
        cp = close.shift(1); vp = v.shift(1)
        if side == "long":
            return (cp < vp) & (close > v) & (df["volume"] > 1.3 * df["volume"].rolling(20).mean())
        return (cp > vp) & (close < v) & (df["volume"] > 1.3 * df["volume"].rolling(20).mean())
    if kind == "engulfing":
        body_prev = close.shift(1) - op.shift(1)
        body_curr = close - op
        if side == "long":
            return (body_prev < 0) & (body_curr > 0) & (close > op.shift(1)) & (op < close.shift(1))
        return (body_prev > 0) & (body_curr < 0) & (close < op.shift(1)) & (op > close.shift(1))
    if kind == "pin_bar":
        body_abs = body.abs()
        lw = np.minimum(close, op) - low; uw = high - np.maximum(close, op)
        if side == "long":
            return (lw > 2 * body_abs) & (lw > 0.55 * rng) & (close > op)
        return (uw > 2 * body_abs) & (uw > 0.55 * rng) & (close < op)
    if kind == "momentum_bar":
        body_ratio = body.abs() / rng
        close_pos = (close - low) / rng
        vol_surge = df["volume"] > 1.5 * df["volume"].rolling(20).mean()
        if side == "long":
            return (body > 0) & (body_ratio > 0.65) & (close_pos > 0.75) & vol_surge
        return (body < 0) & (body_ratio > 0.65) & (close_pos < 0.25) & vol_surge
    if kind == "2bar_reversal":
        # 2-bar reversal: bar -1 strong opposite, bar 0 reverses
        body_prev = close.shift(1) - op.shift(1)
        body_curr = close - op
        if side == "long":
            return (body_prev < -0.4 * rng.shift(1)) & (body_curr > 0.4 * rng) & (close > close.shift(1))
        return (body_prev > 0.4 * rng.shift(1)) & (body_curr < -0.4 * rng) & (close < close.shift(1))
    if kind == "vol_squeeze_breakout":
        # BB width in lowest 25% of last 50 bars, then breakout
        bb_w = (bollinger(close, 20, 2.0)[2] - bollinger(close, 20, 2.0)[1]) / close
        bb_w_q25 = bb_w.rolling(50).quantile(0.25)
        squeeze_recent = (bb_w <= bb_w_q25).rolling(5).max().fillna(False).astype(bool)
        u5 = high.rolling(5).max().shift(1); l5 = low.rolling(5).min().shift(1)
        vol_surge = df["volume"] > 1.35 * df["volume"].rolling(20).mean()
        if side == "long":
            return squeeze_recent & (close > u5) & vol_surge
        return squeeze_recent & (close < l5) & vol_surge
    raise ValueError(kind)


def confirmation(df, kind):
    """Returns boolean series — confirmation filter."""
    close = df["close"]
    if kind == "none":
        return pd.Series(True, index=df.index)
    if kind == "session_us_eu":
        h = pd.Series(df.index.hour, index=df.index)
        return (h >= 12) & (h < 21)
    if kind == "session_asia":
        h = pd.Series(df.index.hour, index=df.index)
        return (h >= 0) & (h < 8)
    if kind == "vol_surge_15":
        return df["volume"] > 1.5 * df["volume"].rolling(20).mean()
    if kind == "vol_surge_20":
        return df["volume"] > 2.0 * df["volume"].rolling(20).mean()
    if kind == "taker_buy":
        return taker_buy_ratio(df) > 0.55
    if kind == "taker_sell":
        return taker_buy_ratio(df) < 0.45
    if kind == "rsi_50_65":
        r = rsi(close, 14); return (r > 50) & (r < 65)
    if kind == "rsi_35_50":
        r = rsi(close, 14); return (r > 35) & (r < 50)
    if kind == "atr_expand":
        a = atr(df, 14); a_ma = a.rolling(14).mean()
        return a > a_ma
    raise ValueError(kind)


def stop_price(df, kind, side):
    """Returns stop price series (signal bar close)."""
    close = df["close"]; low = df["low"]; high = df["high"]
    a = atr(df, 14)
    if kind == "atr_0_5":
        return close - 0.5 * a if side == "long" else close + 0.5 * a
    if kind == "atr_0_75":
        return close - 0.75 * a if side == "long" else close + 0.75 * a
    if kind == "atr_1_0":
        return close - 1.0 * a if side == "long" else close + 1.0 * a
    if kind == "atr_1_5":
        return close - 1.5 * a if side == "long" else close + 1.5 * a
    if kind == "atr_2_0":
        return close - 2.0 * a if side == "long" else close + 2.0 * a
    if kind == "swing_5":
        sl = low.rolling(5).min().shift(1) if side == "long" else high.rolling(5).max().shift(1)
        return sl - 0.1 * a if side == "long" else sl + 0.1 * a
    if kind == "swing_10":
        sl = low.rolling(10).min().shift(1) if side == "long" else high.rolling(10).max().shift(1)
        return sl - 0.1 * a if side == "long" else sl + 0.1 * a
    if kind == "signal_bar_low":
        return low - 0.05 * a if side == "long" else high + 0.05 * a
    if kind == "prev_bar_low":
        return low.shift(1) - 0.1 * a if side == "long" else high.shift(1) + 0.1 * a
    raise ValueError(kind)


# ============================================================================
# 100-strategy generator
# ============================================================================

# Component pools — designed to span the design space
TRENDS = ["ema_9_21", "ema_20_50", "ema_50_200", "htf_ema_200", "htf_strong", "linreg_20", "adx_25", "none"]
ENTRIES = ["breakout_10", "breakout_20", "breakout_50", "pullback_ema20", "pullback_ema50",
           "rsi_reversal", "rsi2_extreme", "bb_touch", "vwap_reclaim", "engulfing",
           "pin_bar", "momentum_bar", "2bar_reversal", "vol_squeeze_breakout"]
CONFIRMS = ["none", "session_us_eu", "session_asia", "vol_surge_15", "vol_surge_20",
            "taker_buy", "taker_sell", "rsi_50_65", "rsi_35_50", "atr_expand"]
STOPS = ["atr_0_5", "atr_0_75", "atr_1_0", "atr_1_5", "atr_2_0",
         "swing_5", "swing_10", "signal_bar_low", "prev_bar_low"]
TARGETS = [1.5, 2.0, 2.5, 3.0]
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["1h", "4h"]
SIDES = ["long", "short", "both"]


def generate_100_strategies(seed: int = 42) -> list[dict]:
    """Generate 100 distinct strategy configurations by sampling from the pools."""
    import random
    rng = random.Random(seed)
    configs = []
    seen = set()

    # Pass 1: deliberately spread across timeframes and symbols
    # 50 strategies on 1h, 50 on 4h. Half BTC, half ETH. Mix of sides.
    attempts = 0
    while len(configs) < 100 and attempts < 10000:
        attempts += 1
        tf = "1h" if len(configs) < 50 else "4h"
        sym = "BTCUSDT" if (len(configs) % 2 == 0) else "ETHUSDT"
        side = rng.choice(SIDES)
        trend = rng.choice(TRENDS)
        entry = rng.choice(ENTRIES)
        confirm = rng.choice(CONFIRMS)
        stop = rng.choice(STOPS)
        target = rng.choice(TARGETS)
        max_bars = 24 if tf == "1h" else 12  # 24h on 1h, 48h on 4h
        key = (sym, tf, side, trend, entry, confirm, stop, target)
        if key in seen:
            continue
        seen.add(key)
        cid = f"BOT_{len(configs)+1:03d}_{sym[:3]}_{tf}_{side[:3]}_{entry[:8]}_{trend[:8]}_R{target}"
        configs.append({
            "id": cid, "symbol": sym, "timeframe": tf, "side": side,
            "trend": trend, "entry": entry, "confirm": confirm,
            "stop": stop, "target": target, "max_bars": max_bars,
        })
    return configs


# ============================================================================
# Backtester (bi-directional, USD wallet)
# ============================================================================

def backtest_strategy(df, htf, cfg) -> tuple[list, float]:
    """Run a single strategy config. Returns (trades_list, final_wallet)."""
    trend_long, trend_short = trend_filter(df, htf, cfg["trend"])
    confirm_mask = confirmation(df, cfg["confirm"])

    # Per-side signals
    if cfg["side"] == "long":
        sig_long = entry_trigger(df, cfg["entry"], "long") & trend_long & confirm_mask
        sig_short = pd.Series(False, index=df.index)
        stop_long = stop_price(df, cfg["stop"], "long")
        stop_short = pd.Series(np.nan, index=df.index)
    elif cfg["side"] == "short":
        sig_short = entry_trigger(df, cfg["entry"], "short") & trend_short & confirm_mask
        sig_long = pd.Series(False, index=df.index)
        stop_short = stop_price(df, cfg["stop"], "short")
        stop_long = pd.Series(np.nan, index=df.index)
    else:  # both
        sig_long = entry_trigger(df, cfg["entry"], "long") & trend_long & confirm_mask
        sig_short = entry_trigger(df, cfg["entry"], "short") & trend_short & confirm_mask
        stop_long = stop_price(df, cfg["stop"], "long")
        stop_short = stop_price(df, cfg["stop"], "short")

    r_mult = cfg["target"]
    max_bars = cfg["max_bars"]

    df = df.sort_index()
    n = len(df)
    if n < 50:
        return [], INITIAL_WALLET

    sig_long_arr = sig_long.fillna(False).values
    sig_short_arr = sig_short.fillna(False).values
    stop_long_arr = stop_long.values
    stop_short_arr = stop_short.values

    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values; closes = df["close"].values
    times = df.index

    slip_frac = COSTS.slippage_bps / 10000.0

    trades = []
    wallet = INITIAL_WALLET
    i = 0
    while i < n:
        signal_long = i+1 < n and sig_long_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_long_arr[i])
        signal_short = i+1 < n and sig_short_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_short_arr[i])
        chosen = None
        if signal_long and signal_short:
            chosen = "long"  # prefer long when both fire
        elif signal_long:
            chosen = "long"
        elif signal_short:
            chosen = "short"
        if chosen is None:
            i += 1; continue

        entry_raw = float(opens[i+1])
        if entry_raw <= 0:
            i += 1; continue
        if chosen == "long":
            stop_price_val = float(stop_long_arr[i])
            if not np.isfinite(stop_price_val) or stop_price_val >= entry_raw:
                i += 1; continue
            entry_price = entry_raw * (1 + slip_frac)
            target_price = entry_price + r_mult * (entry_price - stop_price_val)
        else:
            stop_price_val = float(stop_short_arr[i])
            if not np.isfinite(stop_price_val) or stop_price_val <= entry_raw:
                i += 1; continue
            entry_price = entry_raw * (1 - slip_frac)
            target_price = entry_price - r_mult * (stop_price_val - entry_price)

        risk_per_unit = abs(entry_price - stop_price_val)
        if risk_per_unit <= 0:
            i += 1; continue
        risk_amount = RISK_PER_TRADE * wallet
        size_units = risk_amount / risk_per_unit
        if size_units <= 0:
            i += 1; continue

        entry_notional = size_units * entry_price
        entry_commission = entry_notional * COSTS.commission_pct
        wallet_after_entry = wallet - entry_commission

        exit_idx = -1; exit_price = float("nan"); exit_reason = ""; bars_held = 0
        j = i + 1
        while j < n:
            bars_held = j - i
            h = highs[j]; l = lows[j]
            if chosen == "long":
                hit_stop = l <= stop_price_val
                hit_tgt = h >= target_price
            else:
                hit_stop = h >= stop_price_val
                hit_tgt = l <= target_price
            if hit_stop and hit_tgt:
                exit_price = stop_price_val; exit_reason = "stop"; exit_idx = j; break
            elif hit_stop:
                exit_price = stop_price_val; exit_reason = "stop"; exit_idx = j; break
            elif hit_tgt:
                exit_price = target_price; exit_reason = "target"; exit_idx = j; break
            if bars_held >= max_bars:
                exit_price = closes[j]; exit_reason = "time"; exit_idx = j; break
            j += 1
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

        trades.append({
            "entry_time": times[i+1], "exit_time": times[exit_idx],
            "side": chosen, "entry_price": entry_price, "exit_price": exit_fill,
            "stop": stop_price_val, "target": target_price, "size": size_units,
            "risk_usd": risk_amount, "gross_pnl": gross_pnl, "cost": total_cost,
            "net_pnl": net_pnl, "wallet_before": wallet,
            "wallet_after": wallet_after_entry + net_pnl,
            "R_multiple": r_mult_actual, "exit_reason": exit_reason, "bars_held": bars_held,
        })
        wallet = wallet_after_entry + net_pnl
        i = exit_idx + 1

    return trades, wallet


# ============================================================================
# Excel writer
# ============================================================================

def write_excel(top_10_results, all_100_results, out_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    pos_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    neg_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    warn_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))

    # Sheet 1: All 100 ranked
    ws = wb.create_sheet("All 100 Ranked")
    ws["A1"] = "100 CUSTOM BOTS — 30-DAY LIVE SIMULATION, RANKED BY NET PROFIT"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = f"Window: {START_DATE} → {END_DATE}  |  Initial wallet: ${INITIAL_WALLET:,.2f}  |  Risk: 1%/trade  |  Costs: 0.05%+1bp/side"
    ws["A3"] = f"Total bots tested: {len(all_100_results)}  |  Profitable: {sum(1 for r in all_100_results if r['net']>0)}  |  Breakeven: {sum(1 for r in all_100_results if r['net']==0)}  |  Losing: {sum(1 for r in all_100_results if r['net']<0)}"

    headers = ["Rank", "Bot ID", "Symbol", "TF", "Side", "Trend", "Entry", "Confirm", "Stop", "Target R",
               "Trades", "Wins", "Losses", "Other", "Win Rate", "Initial $", "Final $", "Net $", "Net %", "Max DD %"]
    row_idx = 5
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=row_idx, column=col, value=h)
        c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center"); c.border = border

    sorted_results = sorted(all_100_results, key=lambda x: x["net"], reverse=True)
    for rank, r in enumerate(sorted_results, 1):
        cfg = r["cfg"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["exit_reason"] == "target" and t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["exit_reason"] == "stop" and t["net_pnl"] < 0)
        n_other = len(trades) - n_wins - n_losses
        wr = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0
        wallets = [INITIAL_WALLET] + [t["wallet_after"] for t in trades]
        peak = np.maximum.accumulate(wallets)
        dd = (peak - wallets) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0

        row = [rank, cfg["id"], cfg["symbol"], cfg["timeframe"], cfg["side"],
               cfg["trend"], cfg["entry"], cfg["confirm"], cfg["stop"], cfg["target"],
               len(trades), n_wins, n_losses, n_other, wr,
               INITIAL_WALLET, r["final"], r["net"], r["net_pct"], max_dd]
        for col, v in enumerate(row, 1):
            c = ws.cell(row=row_idx+rank, column=col, value=v)
            c.border = border
            if col in (16, 17, 18):
                c.number_format = '"$"#,##0.00'
            elif col in (19, 20):
                c.number_format = '0.00%'
            elif col == 15:
                c.number_format = '0.0%'
            if col == 18 and isinstance(v, (int, float)):
                c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
    widths = [6, 40, 10, 5, 7, 14, 18, 16, 16, 9, 8, 7, 8, 8, 9, 12, 12, 12, 9, 9]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A6"

    # Sheet 2: Top 10 summary
    ws = wb.create_sheet("Top 10 Summary")
    ws["A1"] = "TOP 10 BOTS BY NET PROFIT — Last 30 Days"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = f"Initial wallet: ${INITIAL_WALLET:,.2f}  |  Risk: 1%/trade  |  Realistic costs applied"
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=4, column=col, value=h)
        c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center"); c.border = border
    for rank, r in enumerate(top_10_results, 1):
        cfg = r["cfg"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["exit_reason"] == "target" and t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["exit_reason"] == "stop" and t["net_pnl"] < 0)
        n_other = len(trades) - n_wins - n_losses
        wr = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0
        wallets = [INITIAL_WALLET] + [t["wallet_after"] for t in trades]
        peak = np.maximum.accumulate(wallets)
        dd = (peak - wallets) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0
        row = [rank, cfg["id"], cfg["symbol"], cfg["timeframe"], cfg["side"],
               cfg["trend"], cfg["entry"], cfg["confirm"], cfg["stop"], cfg["target"],
               len(trades), n_wins, n_losses, n_other, wr,
               INITIAL_WALLET, r["final"], r["net"], r["net_pct"], max_dd]
        for col, v in enumerate(row, 1):
            c = ws.cell(row=4+rank, column=col, value=v)
            c.border = border
            if col in (16, 17, 18):
                c.number_format = '"$"#,##0.00'
            elif col in (19, 20):
                c.number_format = '0.00%'
            elif col == 15:
                c.number_format = '0.0%'
            if col == 18:
                c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A5"

    # Sheets 3-12: Per-strategy trade logs for top 10
    for rank, r in enumerate(top_10_results, 1):
        cfg = r["cfg"]; trades = r["trades"]
        sheet_name = f"#{rank} {cfg['id']}"[:31]
        ws = wb.create_sheet(sheet_name)
        ws["A1"] = f"#{rank} — {cfg['id']}  |  {cfg['symbol']} {cfg['timeframe']} {cfg['side']}  |  Net: ${r['net']:+,.2f} ({r['net_pct']*100:+.2f}%)  |  Trades: {len(trades)}"
        ws["A1"].font = Font(bold=True, size=11)
        ws.merge_cells("A1:R1")
        ws["A2"] = f"Trend={cfg['trend']}  Entry={cfg['entry']}  Confirm={cfg['confirm']}  Stop={cfg['stop']}  Target={cfg['target']}R"
        ws.merge_cells("A2:R2")

        trade_headers = ["#", "Entry Time", "Exit Time", "Side", "Entry $", "Exit $",
                         "Stop $", "Target $", "Size", "Risk $", "Gross P&L $",
                         "Cost $", "Net P&L $", "R Mult", "Exit Reason", "Bars Held",
                         "Wallet Before $", "Wallet After $"]
        for col, h in enumerate(trade_headers, 1):
            c = ws.cell(row=4, column=col, value=h)
            c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center"); c.border = border
        for i, t in enumerate(trades, 1):
            entry_t = t["entry_time"].tz_localize(None) if hasattr(t["entry_time"], "tz_localize") else t["entry_time"]
            exit_t = t["exit_time"].tz_localize(None) if hasattr(t["exit_time"], "tz_localize") else t["exit_time"]
            row = [i, entry_t, exit_t, t["side"], t["entry_price"], t["exit_price"],
                   t["stop"], t["target"], t["size"], t["risk_usd"], t["gross_pnl"],
                   t["cost"], t["net_pnl"], t["R_multiple"], t["exit_reason"],
                   t["bars_held"], t["wallet_before"], t["wallet_after"]]
            for col, v in enumerate(row, 1):
                c = ws.cell(row=4+i, column=col, value=v)
                c.border = border
                if col in (5, 6, 7, 8, 10, 11, 12, 13, 17, 18) and isinstance(v, (int, float)):
                    c.number_format = '"$"#,##0.00'
                elif col == 14 and isinstance(v, (int, float)):
                    c.number_format = '0.00'
                if col == 13 and isinstance(v, (int, float)):
                    c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
        twidths = [5, 22, 22, 6, 12, 12, 12, 12, 12, 10, 12, 10, 12, 8, 12, 8, 14, 14]
        for i, w in enumerate(twidths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A5"

    wb.save(out_path)


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*100)
    print("100 CUSTOM BOTS — 30-DAY LIVE SIMULATION")
    print(f"Window: {START_DATE} → {END_DATE}  |  Initial: ${INITIAL_WALLET:,.2f}  |  Risk: 1%/trade")
    print("="*100)

    configs = generate_100_strategies(seed=42)
    print(f"\nGenerated {len(configs)} distinct strategy configurations")

    # Pre-load all data
    print("\nLoading data...")
    data_cache = {}
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            data_cache[(sym, tf)] = load_window(sym, tf)
            print(f"  {sym} {tf}: {len(data_cache[(sym,tf)][0])} bars (incl. warmup)")

    # Run all 100
    print("\nRunning 100 bots on last 30 days...")
    all_results = []
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        df, htf = data_cache[(cfg["symbol"], cfg["timeframe"])]
        try:
            trades, final = backtest_strategy(df, htf, cfg)
            net = final - INITIAL_WALLET
            net_pct = net / INITIAL_WALLET
            all_results.append({"cfg": cfg, "trades": trades, "final": final, "net": net, "net_pct": net_pct})
            n_wins = sum(1 for t in trades if t["exit_reason"] == "target" and t["net_pnl"] > 0)
            n_losses = sum(1 for t in trades if t["exit_reason"] == "stop" and t["net_pnl"] < 0)
            wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
            tag = "✓" if net > 0 else ("·" if net == 0 else "✗")
            print(f"  {tag} {cfg['id']:50s} {len(trades):>3}t  WR={wr*100:>5.1f}%  net=${net:>+9,.2f} ({net_pct*100:>+6.2f}%)")
        except Exception as e:
            import traceback; print(f"  ✗ {cfg['id']} FAILED: {e}"); traceback.print_exc()
            all_results.append({"cfg": cfg, "trades": [], "final": INITIAL_WALLET, "net": 0, "net_pct": 0})
    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Rank and pick top 10
    all_results.sort(key=lambda x: x["net"], reverse=True)
    top_10 = all_results[:10]

    # Print summary
    n_profit = sum(1 for r in all_results if r["net"] > 0)
    n_break = sum(1 for r in all_results if r["net"] == 0)
    n_loss = sum(1 for r in all_results if r["net"] < 0)
    print("\n" + "="*100)
    print(f"RESULTS — 100 bots tested on last 30 days")
    print(f"  Profitable: {n_profit}  |  Breakeven (no trades): {n_break}  |  Losing: {n_loss}")
    print(f"  Best: {top_10[0]['cfg']['id']} ${top_10[0]['net']:>+,.2f} ({top_10[0]['net_pct']*100:+.2f}%)")
    print(f"  Worst: {all_results[-1]['cfg']['id']} ${all_results[-1]['net']:>+,.2f} ({all_results[-1]['net_pct']*100:+.2f}%)")
    print(f"  Median: ${all_results[50]['net']:>+,.2f} ({all_results[50]['net_pct']*100:+.2f}%)")

    print("\n" + "="*100)
    print("TOP 10 BY NET PROFIT")
    print("="*100)
    print(f"{'#':<3} {'Bot ID':<50} {'Sym':<8} {'TF':<4} {'Side':<6} {'Trades':>6} {'WR':>6} {'Net $':>11} {'Net %':>8}")
    print("-"*100)
    for rank, r in enumerate(top_10, 1):
        cfg = r["cfg"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["exit_reason"] == "target" and t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["exit_reason"] == "stop" and t["net_pnl"] < 0)
        wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
        print(f"{rank:<3} {cfg['id'][:50]:<50} {cfg['symbol']:<8} {cfg['timeframe']:<4} {cfg['side']:<6} {len(trades):>6} {wr*100:>5.1f}% {r['net']:>+11,.2f} {r['net_pct']*100:>+7.2f}%")

    # Write Excel
    out_path = DOWNLOAD / "100_bots_30day.xlsx"
    write_excel(top_10, all_results, out_path)
    print(f"\n>>> Wrote {out_path}")
    print(f"    Sheets: 'All 100 Ranked' + 'Top 10 Summary' + 10 per-strategy trade logs")


if __name__ == "__main__":
    main()
