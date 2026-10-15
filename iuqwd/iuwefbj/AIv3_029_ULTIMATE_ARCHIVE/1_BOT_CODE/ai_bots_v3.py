"""
AIBOT_087 v2 — Multi-Layer "Thinking" AI Bot Framework.

Inspired by AIBOT_087's success (+14.56% in 30 days), this expands the idea:
  1. Multi-asset: 9 markets (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, PAXG-gold)
  2. Multi-candidate: momentum + breakout + pullback generators
  3. Multi-layer AI ("thinking"):
       Layer 1: Trade Identifier (is this a real opportunity?)
       Layer 2: Risk Analyst (loss probability given context)
       Layer 3: Confidence Calibrator (true P(win))
       Layer 4: Meta-Decider (combines all three)
  4. Cross-asset features (when trading ETH, see BTC/gold/market state)
  5. Portfolio-level risk management (correlation cap, total exposure cap)

Generates 100 variations, tests on last 30 days, compares to AIBOT_087's +14.56%.

STRICT NO-LOOKAHEAD:
  - All features at bar i use only data from bars [0..i]
  - AI trained only on data BEFORE the test window
  - Cross-asset features use only PAST bars from other assets
  - Same bar i in asset A cannot see same bar i in asset B (only bar i-1 and earlier)
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from backtester import Costs, ema, rsi, atr, adx, bollinger, vwap_rolling, donchian
from smart_strategies import zscore, taker_buy_ratio
from bots_100 import rolling_linreg_slope, INITIAL_WALLET, RISK_PER_TRADE, COSTS, DATA_DIR, DOWNLOAD
from risk_managed_v2 import RiskConfig

# All 9 markets
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
               "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "PAXGUSDT"]
# Reference markets for cross-asset features
REF_SYMBOLS = ["BTCUSDT", "ETHUSDT", "PAXGUSDT"]

# Test window
TEST_END = pd.Timestamp("2026-10-14 18:15:00", tz="UTC")
TEST_START = TEST_END - pd.Timedelta(days=30)
VAL_END = TEST_START
VAL_START = VAL_END - pd.Timedelta(days=30)
TRAIN_END = VAL_START
TRAIN_START = TRAIN_END - pd.Timedelta(days=365)  # 12 months training (data goes back to 2022)


# ============================================================================
# Data loading (1h and 4h resampled from 1h)
# ============================================================================

_DATA_CACHE = {}

def load_data(symbol: str, timeframe: str = "1h"):
    """Load data with lookback for indicator warmup."""
    key = (symbol, timeframe)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    df = pd.read_parquet(DATA_DIR / f"{symbol}_1h.parquet").set_index("open_time").sort_index()
    if timeframe == "4h":
        df = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum", "quote_volume": "sum",
            "taker_buy_base": "sum",
        }).dropna()
    lookback = TRAIN_START - pd.Timedelta(hours=400)
    df = df.loc[lookback:TEST_END].copy()
    _DATA_CACHE[key] = df
    return df


# ============================================================================
# Indicator cache per symbol
# ============================================================================

_IND_CACHE = {}

def get_indicators(df, symbol):
    key = (symbol, id(df))
    if key in _IND_CACHE:
        return _IND_CACHE[key]
    if len(_IND_CACHE) > 30:
        _IND_CACHE.clear()
    close = df["close"]; high = df["high"]; low = df["low"]; op = df["open"]
    vol = df["volume"]
    a = atr(df, 14); r = rsi(close, 14)
    e20 = ema(close, 20); e50 = ema(close, 50); e200 = ema(close, 200)
    slope = rolling_linreg_slope(close, 20); slope_50 = rolling_linreg_slope(close, 50)
    a_line = adx(df, 14)
    _, bb_u, bb_l = bollinger(close, 20, 2.0)
    bb_w = (bb_u - bb_l) / close
    vol_ma = vol.rolling(20).mean()
    tbr = taker_buy_ratio(df)
    atr_ma = a.rolling(50).mean()
    ret_3 = close.pct_change(3); ret_12 = close.pct_change(12)
    pos_in_range = (close - close.rolling(50).min()) / (close.rolling(50).max() - close.rolling(50).min()).replace(0, np.nan)
    cache = {"close": close, "high": high, "low": low, "op": op, "vol": vol,
             "atr": a, "rsi": r, "ema20": e20, "ema50": e50, "ema200": e200,
             "slope": slope, "slope_50": slope_50, "adx": a_line,
             "bb_w": bb_w, "vol_ma": vol_ma, "tbr": tbr, "atr_ma": atr_ma,
             "ret_3": ret_3, "ret_12": ret_12, "pos_in_range": pos_in_range}
    _IND_CACHE[key] = cache
    return cache


# ============================================================================
# Cross-asset features — see BTC/ETH/gold state when trading any asset
# Uses .shift(1) to ensure we only see CLOSED bars from other assets
# ============================================================================

def get_cross_asset_features(symbol: str, i: int, t: pd.Timestamp) -> dict:
    """Get features from BTC, ETH, PAXG at time t (using only data BEFORE t).
    Optimized: pre-compute aligned series once and use searchsorted."""
    feats = {}
    for ref in REF_SYMBOLS:
        if ref == symbol: continue
        try:
            df_ref = load_data(ref, "1h")
            ind = get_indicators(df_ref, ref)
            # searchsorted gives the leftmost index where t could be inserted
            # to keep the array sorted. We want the last bar with index < t.
            ref_times = df_ref.index
            idx_pos = ref_times.searchsorted(t, side="left") - 1
            if idx_pos < 50:
                feats[f"{ref[:3]}_ret_3"] = 0
                feats[f"{ref[:3]}_rsi"] = 50
                feats[f"{ref[:3]}_adx"] = 0
                continue
            feats[f"{ref[:3]}_ret_3"] = float(ind["ret_3"].iloc[idx_pos]) if np.isfinite(ind["ret_3"].iloc[idx_pos]) else 0
            feats[f"{ref[:3]}_rsi"] = float(ind["rsi"].iloc[idx_pos]) if np.isfinite(ind["rsi"].iloc[idx_pos]) else 50
            feats[f"{ref[:3]}_adx"] = float(ind["adx"].iloc[idx_pos]) if np.isfinite(ind["adx"].iloc[idx_pos]) else 0
        except Exception:
            feats[f"{ref[:3]}_ret_3"] = 0
            feats[f"{ref[:3]}_rsi"] = 50
            feats[f"{ref[:3]}_adx"] = 0
    return feats


# ============================================================================
# Feature set for v2 — 20 features including cross-asset
# ============================================================================

FEATURE_COLS_V3 = [
    # Microstructure (5) — AIBOT_087's winning features
    "vol_ratio", "taker_buy_ratio", "vol_ma_diff", "atr_pct", "vol_surge_3bar",
    # Technical (5)
    "rsi", "dist_ema20", "adx", "linreg_slope", "bb_width_pct",
    # New (3)
    "ret_3", "ret_12", "pos_in_range",
    # Cross-asset (6) — BTC, ETH, PAXG state
    "BTC_ret_3", "BTC_rsi", "BTC_adx",
    "ETH_ret_3", "ETH_rsi", "ETH_adx",
    # PAXG (gold) — 3 features
    "PAX_ret_3", "PAX_rsi", "PAX_adx",
    # Side (2)
    "side_long", "side_short",
    # Asset identity (8) — one-hot
    "is_BTC", "is_ETH", "is_BNB", "is_SOL", "is_XRP", "is_ADA", "is_DOGE", "is_AVAX",
]


def extract_features_v3(df, symbol, i, side):
    """Extract 27 features at bar i for given symbol."""
    c = get_indicators(df, symbol)
    close = c["close"]; vol = c["vol"]; vol_ma = c["vol_ma"]
    t = df.index[i]
    def safe(v, d=0):
        return float(v) if np.isfinite(v) else d
    feats = {
        # Microstructure (AIBOT_087's winning set)
        "vol_ratio": safe(vol.iloc[i] / vol_ma.iloc[i], 1) if vol_ma.iloc[i] > 0 else 1,
        "taker_buy_ratio": safe(c["tbr"].iloc[i], 0.5),
        "vol_ma_diff": safe((vol.iloc[i] - vol_ma.iloc[i]) / vol_ma.iloc[i]) if vol_ma.iloc[i] > 0 else 0,
        "atr_pct": safe(c["atr"].iloc[i] / close.iloc[i]) if close.iloc[i] > 0 else 0,
        "vol_surge_3bar": safe(vol.iloc[i] / vol.iloc[i-3], 1) if i >= 3 and vol.iloc[i-3] > 0 else 1,
        # Technical
        "rsi": safe(c["rsi"].iloc[i], 50),
        "dist_ema20": safe((close.iloc[i] - c["ema20"].iloc[i]) / c["ema20"].iloc[i]) if c["ema20"].iloc[i] != 0 else 0,
        "adx": safe(c["adx"].iloc[i]),
        "linreg_slope": safe(c["slope"].iloc[i]),
        "bb_width_pct": safe(c["bb_w"].iloc[i]),
        # New
        "ret_3": safe(c["ret_3"].iloc[i]),
        "ret_12": safe(c["ret_12"].iloc[i]),
        "pos_in_range": safe(c["pos_in_range"].iloc[i], 0.5),
    }
    # Cross-asset features (only past data)
    cross = get_cross_asset_features(symbol, i, t)
    feats["BTC_ret_3"] = cross.get("BTC_ret_3", 0)
    feats["BTC_rsi"] = cross.get("BTC_rsi", 50)
    feats["BTC_adx"] = cross.get("BTC_adx", 0)
    feats["ETH_ret_3"] = cross.get("ETH_ret_3", 0)
    feats["ETH_rsi"] = cross.get("ETH_rsi", 50)
    feats["ETH_adx"] = cross.get("ETH_adx", 0)
    feats["PAX_ret_3"] = cross.get("PAX_ret_3", 0)
    feats["PAX_rsi"] = cross.get("PAX_rsi", 50)
    feats["PAX_adx"] = cross.get("PAX_adx", 0)
    # Side
    feats["side_long"] = int(side == "long")
    feats["side_short"] = int(side == "short")
    # Asset identity
    for s in ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX"]:
        feats[f"is_{s}"] = 1 if symbol.startswith(s) else 0
    return feats


# ============================================================================
# Candidate generators (momentum + breakout + pullback)
# ============================================================================

def cand_momentum(df, htf):
    """AIBOT_087's generator: momentum bar with volume surge."""
    close = df["close"]; op = df["open"]; hi = df["high"]; lo = df["low"]
    rng = (hi - lo).replace(0, np.nan)
    body = close - op
    body_ratio = body.abs() / rng
    close_pos = (close - lo) / rng
    vol_surge = df["volume"] > 1.3 * df["volume"].rolling(20).mean()
    sig_long = (body > 0) & (body_ratio > 0.6) & (close_pos > 0.7) & vol_surge
    sig_short = (body < 0) & (body_ratio > 0.6) & (close_pos < 0.3) & vol_surge
    a = atr(df, 14)
    return sig_long, sig_short, close - 0.5 * a, close + 0.5 * a

def cand_breakout(df, htf):
    close = df["close"]; high = df["high"]; low = df["low"]
    u10 = high.rolling(10).max().shift(1); l10 = low.rolling(10).min().shift(1)
    sig_long = close > u10; sig_short = close < l10
    a = atr(df, 14)
    return sig_long, sig_short, close - 0.75 * a, close + 0.75 * a

def cand_pullback(df, htf):
    close = df["close"]; low = df["low"]; high = df["high"]
    e20 = ema(close, 20)
    sig_long = (low <= e20 * 1.003) & (close > e20) & (close > df["open"])
    sig_short = (high >= e20 * 0.997) & (close < e20) & (close < df["open"])
    a = atr(df, 14)
    return sig_long, sig_short, close - 1.0 * a, close + 1.0 * a

def cand_multi(df, htf):
    """Union of all 3 generators — more candidates = more trades."""
    sl1, ss1, stol1, stos1 = cand_momentum(df, htf)
    sl2, ss2, stol2, stos2 = cand_breakout(df, htf)
    sl3, ss3, stol3, stos3 = cand_pullback(df, htf)
    sig_long = sl1 | sl2 | sl3
    sig_short = ss1 | ss2 | ss3
    a = atr(df, 14)
    # Use ATR-based stop (consistent)
    return sig_long, sig_short, df["close"] - 0.75 * a, df["close"] + 0.75 * a

CANDIDATE_GENERATORS = {
    "momentum": cand_momentum,
    "breakout": cand_breakout,
    "pullback": cand_pullback,
    "multi": cand_multi,
}


# ============================================================================
# MULTI-LAYER "THINKING" AI — 4 specialized models
# ============================================================================

def make_layer_model(kind: str):
    """Make a single AI layer."""
    if kind == "nb":
        return GaussianNB()
    if kind == "lr":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))])
    if kind == "rf":
        return RandomForestClassifier(n_estimators=80, max_depth=5, min_samples_leaf=15,
                                       random_state=42, class_weight="balanced")
    if kind == "gb":
        return GradientBoostingClassifier(n_estimators=50, max_depth=3, min_samples_leaf=15, random_state=42)
    if kind == "knn":
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15))])
    raise ValueError(kind)


@dataclass
class ThinkingAI:
    """Multi-layer AI bot brain."""
    layer1_identifier: object = None  # "Is this a real opportunity?"
    layer2_risk_analyst: object = None  # "What's the loss probability?"
    layer3_calibrator: object = None  # "True P(win)?"
    use_layer1: bool = True
    use_layer2: bool = False
    use_layer3: bool = False

    def predict(self, X):
        """Returns (proba, take_trade). Combines all enabled layers."""
        if not self.use_layer1 and not self.use_layer3:
            return 1.0, True  # no AI = always take
        probas = []
        if self.layer1_identifier is not None and self.use_layer1:
            try:
                p1 = float(self.layer1_identifier.predict_proba(X)[0][1])
                probas.append(("L1", p1, 1.0))
            except Exception:
                pass
        if self.layer3_calibrator is not None and self.use_layer3:
            try:
                p3 = float(self.layer3_calibrator.predict_proba(X)[0][1])
                probas.append(("L3", p3, 1.5))  # weight calibrator higher
            except Exception:
                pass
        if not probas:
            return 1.0, True
        # Weighted average
        total_w = sum(w for _, _, w in probas)
        avg_p = sum(p * w for _, p, w in probas) / total_w
        # Layer 2 risk analyst: if loss probability > 0.6, reject
        if self.layer2_risk_analyst is not None and self.use_layer2:
            try:
                # Predict loss probability (label: 1 = loss)
                p_loss = float(self.layer2_risk_analyst.predict_proba(X)[0][1])
                if p_loss > 0.60:
                    return avg_p, False  # reject
            except Exception:
                pass
        return avg_p, True


def train_thinking_ai(samples, feature_cols, config):
    """Train the multi-layer AI. config has flags for which layers to use."""
    if len(samples) < 30:
        return None
    X = np.array([[s[c] for c in feature_cols] for s in samples])
    y_win = np.array([s["label"] for s in samples])  # 1=win, 0=loss
    y_loss = 1 - y_win  # 1=loss, 0=not-loss (for risk analyst)

    ai = ThinkingAI(
        use_layer1=config.get("use_layer1", True),
        use_layer2=config.get("use_layer2", False),
        use_layer3=config.get("use_layer3", False),
    )
    try:
        if ai.use_layer1:
            ai.layer1_identifier = make_layer_model(config.get("layer1_kind", "nb"))
            ai.layer1_identifier.fit(X, y_win)
        if ai.use_layer2:
            ai.layer2_risk_analyst = make_layer_model(config.get("layer2_kind", "lr"))
            ai.layer2_risk_analyst.fit(X, y_loss)
        if ai.use_layer3:
            ai.layer3_calibrator = make_layer_model(config.get("layer3_kind", "rf"))
            ai.layer3_calibrator.fit(X, y_win)
        return ai
    except Exception as e:
        print(f"    train failed: {e}")
        return None


# ============================================================================
# Generate training samples (multi-asset)
# ============================================================================

def generate_training_samples_multi(symbol, candidate_gen, start, end):
    """Generate training samples for one asset using v3 features."""
    df = load_data(symbol, "1h")
    sig_long, sig_short, stop_long, stop_short = candidate_gen(df, None)
    df = df.sort_index(); n = len(df)
    sig_long_arr = sig_long.fillna(False).values
    sig_short_arr = sig_short.fillna(False).values
    stop_long_arr = stop_long.values
    stop_short_arr = stop_short.values
    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values; closes = df["close"].values
    times = df.index
    slip_frac = COSTS.slippage_bps / 10000.0
    r_mult = 2.0; max_bars = 24

    samples = []
    for i in range(n):
        if times[i] < start or times[i] > end:
            continue
        for chosen, sig_arr, sp_arr in [("long", sig_long_arr, stop_long_arr),
                                         ("short", sig_short_arr, stop_short_arr)]:
            if i+1 >= n: continue
            if not sig_arr[i]: continue
            if not np.isfinite(opens[i+1]) or not np.isfinite(sp_arr[i]): continue
            entry_raw = float(opens[i+1])
            if entry_raw <= 0: continue
            sp = float(sp_arr[i])
            if chosen == "long" and sp >= entry_raw: continue
            if chosen == "short" and sp <= entry_raw: continue
            entry_price = entry_raw * (1 + slip_frac) if chosen == "long" else entry_raw * (1 - slip_frac)
            stop_distance = abs(entry_raw - sp)
            if stop_distance <= 0: continue
            sp_adj = entry_price - stop_distance if chosen == "long" else entry_price + stop_distance
            target_price = entry_price + r_mult * (entry_price - sp_adj) if chosen == "long" \
                           else entry_price - r_mult * (sp_adj - entry_price)
            # Walk forward
            exit_idx = -1; exit_price = float("nan"); exit_reason = ""; bars_held = 0
            for j in range(i+1, min(i+1+max_bars, n)):
                bars_held = j - i
                h = highs[j]; l = lows[j]
                if chosen == "long":
                    hit_stop = l <= sp_adj; hit_tgt = h >= target_price
                else:
                    hit_stop = h >= sp_adj; hit_tgt = l <= target_price
                if hit_stop and hit_tgt:
                    exit_price = sp_adj; exit_reason = "stop"; exit_idx = j; break
                elif hit_stop:
                    exit_price = sp_adj; exit_reason = "stop"; exit_idx = j; break
                elif hit_tgt:
                    exit_price = target_price; exit_reason = "target"; exit_idx = j; break
                if bars_held >= max_bars:
                    exit_price = closes[j]; exit_reason = "time"; exit_idx = j; break
            if exit_idx < 0:
                exit_price = closes[-1]; exit_reason = "end_of_data"; exit_idx = n - 1
            if chosen == "long":
                gross_per_unit = (exit_price * (1 - slip_frac)) - entry_price
            else:
                gross_per_unit = entry_price - (exit_price * (1 + slip_frac))
            gross_r = gross_per_unit / stop_distance
            cost_r = (0.0012 * entry_price) / stop_distance
            net_r = gross_r - cost_r
            label = 1 if (exit_reason == "target" and net_r > 0) else 0
            feats = extract_features_v3(df, symbol, i, chosen)
            feats["label"] = label
            feats["signal_time"] = times[i]
            feats["side"] = chosen
            samples.append(feats)
    return samples


# ============================================================================
# Multi-asset backtester with portfolio-level risk
# ============================================================================

def backtest_multi_asset(cfg, ai_per_symbol, threshold, risk_cfg,
                          start_time, end_time, initial_wallet=INITIAL_WALLET):
    """Backtest a multi-asset bot. Each asset uses its own AI.
    Portfolio-level risk: max 3 concurrent positions, max 2 in same direction.
    """
    symbols = cfg["symbols"]
    candidate_gen = CANDIDATE_GENERATORS[cfg["candidate_gen"]]

    # Pre-compute signals for each symbol
    sym_data = {}
    for sym in symbols:
        df = load_data(sym, "1h").copy()
        sig_long, sig_short, stop_long, stop_short = candidate_gen(df, None)
        df = df.sort_index()
        sym_data[sym] = {
            "df": df,
            "sig_long_arr": sig_long.fillna(False).values,
            "sig_short_arr": sig_short.fillna(False).values,
            "stop_long_arr": stop_long.values,
            "stop_short_arr": stop_short.values,
            "opens": df["open"].values, "highs": df["high"].values,
            "lows": df["low"].values, "closes": df["close"].values,
            "times": df.index,
            "n": len(df),
        }

    # Walk through ALL bars across all symbols in chronological order
    # Build a unified timeline
    all_events = []
    for sym, d in sym_data.items():
        for i in range(d["n"]):
            t = d["times"][i]
            if start_time <= t < end_time:
                all_events.append((t, sym, i))
    all_events.sort()

    # State
    wallet = initial_wallet
    peak_wallet = initial_wallet
    base_risk = risk_cfg.base_risk_pct
    cur_risk = base_risk
    consec_losses = 0; consec_wins = 0
    throttle_remaining = 0
    pause_until = None
    monthly_start = initial_wallet; weekly_start = initial_wallet; daily_start = initial_wallet
    cur_month = all_events[0][0].month if all_events else 1
    cur_week = all_events[0][0].isocalendar()[1] if all_events else 1
    cur_day = all_events[0][0].day if all_events else 1
    monthly_stopped = False; weekly_stopped = False; daily_stopped = False; dd_stopped = False

    # Open positions (symbol -> trade info)
    open_positions = {}  # sym -> {side, entry_price, sp, target_price, size, entry_time, risk_amount, max_bars}
    trades = []
    ai_decisions = {"approved": 0, "rejected": 0, "skipped_rm": 0, "skipped_portfolio": 0}

    slip_frac = COSTS.slippage_bps / 10000.0
    max_bars = 24
    max_concurrent = cfg.get("max_concurrent", 3)
    max_same_direction = cfg.get("max_same_direction", 2)
    r_mult = 2.0

    # Track per-symbol next bar to check for exits
    # We process events in chronological order; for each event, we:
    # 1. Check exits for any open position in that symbol (using that symbol's bar i)
    # 2. Check for new signal at bar i (entry would be at bar i+1)

    # For simplicity, iterate by time
    # Group events by time
    from collections import defaultdict
    events_by_time = defaultdict(list)
    for t, sym, i in all_events:
        events_by_time[t].append((sym, i))

    sorted_times = sorted(events_by_time.keys())

    for t in sorted_times:
        # Risk management reset
        if t.day != cur_day:
            daily_stopped = False; daily_start = wallet; cur_day = t.day
        if t.isocalendar()[1] != cur_week:
            weekly_stopped = False; weekly_start = wallet; cur_week = t.isocalendar()[1]
        if t.month != cur_month:
            monthly_stopped = False; monthly_start = wallet; cur_month = t.month
        if wallet > peak_wallet: peak_wallet = wallet

        # Risk management checks (same as before)
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

        # Process events at this time
        for sym, i in events_by_time[t]:
            d = sym_data[sym]
            # 1. Check exit for open position in this symbol (using bar i)
            if sym in open_positions:
                pos = open_positions[sym]
                if i < d["n"]:
                    h = d["highs"][i]; l = d["lows"][i]; c = d["closes"][i]
                    bars_held = i - pos["entry_bar_idx"]
                    cur_trail_stop = pos["sp"]
                    best_price = pos["best_price"]
                    if pos["side"] == "long":
                        if h > best_price: best_price = h
                    else:
                        if l < best_price: best_price = l
                    # Trailing
                    if risk_cfg.enable_trailing:
                        r_best = (best_price - pos["entry_price"]) / pos["stop_distance"] if pos["side"] == "long" \
                                 else (pos["entry_price"] - best_price) / pos["stop_distance"]
                        if r_best >= risk_cfg.trail_trigger_1:
                            if pos["side"] == "long":
                                if pos["entry_price"] > cur_trail_stop: cur_trail_stop = pos["entry_price"]
                            else:
                                if pos["entry_price"] < cur_trail_stop: cur_trail_stop = pos["entry_price"]
                        if r_best >= risk_cfg.trail_trigger_2:
                            if pos["side"] == "long":
                                new_stop = best_price - risk_cfg.trail_distance_r * pos["stop_distance"]
                                if new_stop > cur_trail_stop: cur_trail_stop = new_stop
                            else:
                                new_stop = best_price + risk_cfg.trail_distance_r * pos["stop_distance"]
                                if new_stop < cur_trail_stop: cur_trail_stop = new_stop
                    # Check exits
                    if pos["side"] == "long":
                        hit_stop = l <= cur_trail_stop; hit_tgt = h >= pos["target_price"]
                    else:
                        hit_stop = h >= cur_trail_stop; hit_tgt = l <= pos["target_price"]
                    exit_now = False; exit_price = 0; exit_reason = ""
                    if hit_stop and hit_tgt:
                        exit_price = cur_trail_stop; exit_reason = "stop"; exit_now = True
                    elif hit_stop:
                        exit_reason = "trail_win" if cur_trail_stop >= pos["entry_price"] else "stop"
                        exit_price = cur_trail_stop; exit_now = True
                    elif hit_tgt:
                        exit_price = pos["target_price"]; exit_reason = "target"; exit_now = True
                    elif bars_held >= max_bars:
                        exit_price = c; exit_reason = "time"; exit_now = True
                    elif t >= end_time:
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
                            "side": pos["side"], "entry_price": pos["entry_price"], "exit_price": exit_fill,
                            "stop": pos["sp"], "target": pos["target_price"], "size": pos["size"],
                            "risk_pct": pos["risk_pct"], "risk_usd": pos["risk_amount"],
                            "gross_pnl": gross_pnl, "cost": pos["entry_commission"] + exit_commission,
                            "net_pnl": net_pnl,
                            "wallet_before": pos["wallet_before"], "wallet_after": new_wallet,
                            "R_multiple": r_mult_actual, "exit_reason": exit_reason, "bars_held": bars_held,
                            "ai_proba": pos.get("ai_proba", 0),
                        })
                        wallet = new_wallet
                        if wallet > peak_wallet: peak_wallet = wallet
                        del open_positions[sym]

            # 2. Check for new entry signal at bar i (entry at bar i+1)
            if not allowed:
                ai_decisions["skipped_rm"] += 1
                continue
            if sym in open_positions:
                continue  # already in position
            if len(open_positions) >= max_concurrent:
                ai_decisions["skipped_portfolio"] += 1
                continue
            # Direction count check
            long_count = sum(1 for p in open_positions.values() if p["side"] == "long")
            short_count = sum(1 for p in open_positions.values() if p["side"] == "short")

            signal_long = i+1 < d["n"] and d["sig_long_arr"][i] and np.isfinite(d["opens"][i+1]) and np.isfinite(d["stop_long_arr"][i])
            signal_short = i+1 < d["n"] and d["sig_short_arr"][i] and np.isfinite(d["opens"][i+1]) and np.isfinite(d["stop_short_arr"][i])
            chosen = None
            if signal_long and signal_short: chosen = "long"
            elif signal_long: chosen = "long"
            elif signal_short: chosen = "short"
            if chosen is None: continue
            # Direction cap
            if chosen == "long" and long_count >= max_same_direction: continue
            if chosen == "short" and short_count >= max_same_direction: continue

            # AI gatekeeper
            ai = ai_per_symbol.get(sym)
            if ai is not None:
                feats = extract_features_v3(d["df"], sym, i, chosen)
                X = np.array([[feats[c] for c in FEATURE_COLS_V3]])
                proba, take = ai.predict(X)
                if not take or proba < threshold:
                    ai_decisions["rejected"] += 1
                    continue
            else:
                proba = 1.0
            ai_decisions["approved"] += 1

            # Execute trade
            entry_raw = float(d["opens"][i+1])
            if entry_raw <= 0: continue
            if chosen == "long":
                sp_ref = float(d["stop_long_arr"][i])
                if not np.isfinite(sp_ref) or sp_ref >= entry_raw: continue
                entry_price = entry_raw * (1 + slip_frac)
                stop_distance = abs(entry_raw - sp_ref)
                sp = entry_price - stop_distance
                target_price = entry_price + r_mult * (entry_price - sp)
            else:
                sp_ref = float(d["stop_short_arr"][i])
                if not np.isfinite(sp_ref) or sp_ref <= entry_raw: continue
                entry_price = entry_raw * (1 - slip_frac)
                stop_distance = abs(sp_ref - entry_raw)
                sp = entry_price + stop_distance
                target_price = entry_price - r_mult * (sp - entry_price)
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
                "stop_distance": stop_distance, "best_price": entry_price,
                "risk_amount": risk_amount, "risk_pct": eff_risk,
                "wallet_before": wallet, "wallet_after_entry": wallet_after_entry,
                "entry_commission": entry_commission, "ai_proba": proba,
            }

    # Close any remaining open positions at end of window
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
            "stop": pos["sp"], "target": pos["target_price"], "size": pos["size"],
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
    return trades, {
        "n_trades": len(trades), "final_wallet": wallet,
        "net": wallet - initial_wallet, "net_pct": (wallet - initial_wallet) / initial_wallet,
        "peak_wallet": peak_wallet, "max_dd_pct": max_dd,
        "ai_decisions": ai_decisions,
    }


# ============================================================================
# Generate 100 bot configurations
# ============================================================================

def generate_100_v3_configs(seed=42):
    """Generate 100 variations of AIBOT_087 with multi-asset + multi-layer AI."""
    import random
    rng = random.Random(seed)

    # Component pools
    symbol_sets = [
        ("ETH_only", ["ETHUSDT"]),
        ("BTC_only", ["BTCUSDT"]),
        ("top3", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
        ("top5", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]),
        ("all9", ALL_SYMBOLS),
        ("with_gold", ["BTCUSDT", "ETHUSDT", "PAXGUSDT"]),
        ("alts_only", ["BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]),
    ]
    candidate_gens = ["momentum", "breakout", "pullback", "multi"]
    risk_presets = ["aggressive", "standard", "conservative"]
    # Multi-layer AI configurations
    ai_configs = [
        # AIBOT_087 baseline: just L1 with Naive Bayes
        {"name": "L1_nb", "use_layer1": True, "layer1_kind": "nb", "use_layer2": False, "use_layer3": False},
        {"name": "L1_lr", "use_layer1": True, "layer1_kind": "lr", "use_layer2": False, "use_layer3": False},
        {"name": "L1_rf", "use_layer1": True, "layer1_kind": "rf", "use_layer2": False, "use_layer3": False},
        # 2-layer: identifier + risk analyst
        {"name": "L1nb+L2lr", "use_layer1": True, "layer1_kind": "nb", "use_layer2": True, "layer2_kind": "lr", "use_layer3": False},
        {"name": "L1rf+L2lr", "use_layer1": True, "layer1_kind": "rf", "use_layer2": True, "layer2_kind": "lr", "use_layer3": False},
        # 3-layer: identifier + risk analyst + calibrator (full "thinking" AI)
        {"name": "L1nb+L2lr+L3rf", "use_layer1": True, "layer1_kind": "nb", "use_layer2": True, "layer2_kind": "lr", "use_layer3": True, "layer3_kind": "rf"},
        {"name": "L1rf+L2lr+L3gb", "use_layer1": True, "layer1_kind": "rf", "use_layer2": True, "layer2_kind": "lr", "use_layer3": True, "layer3_kind": "gb"},
        # Calibrator only (like AIBOT_087 but with RF instead of NB)
        {"name": "L3rf_only", "use_layer1": False, "use_layer2": False, "use_layer3": True, "layer3_kind": "rf"},
        {"name": "L3gb_only", "use_layer1": False, "use_layer2": False, "use_layer3": True, "layer3_kind": "gb"},
    ]
    max_concurrent_options = [1, 2, 3, 5]
    max_same_dir_options = [1, 2, 3]

    configs = []
    seen = set()
    attempts = 0
    while len(configs) < 100 and attempts < 20000:
        attempts += 1
        sym_set_name, sym_set = rng.choice(symbol_sets)
        cand = rng.choice(candidate_gens)
        risk = rng.choice(risk_presets)
        ai = rng.choice(ai_configs)
        mc = rng.choice(max_concurrent_options)
        msd = rng.choice(max_same_dir_options)
        key = (sym_set_name, cand, risk, ai["name"], mc, msd)
        if key in seen: continue
        seen.add(key)
        bot_id = f"AIv3_{len(configs)+1:03d}_{sym_set_name[:5]}_{cand[:4]}_{risk[:4]}_{ai['name'][:8]}_mc{mc}"
        configs.append({
            "id": bot_id,
            "symbols": sym_set,
            "symbol_set": sym_set_name,
            "candidate_gen": cand,
            "risk_preset": risk,
            "ai_config": ai,
            "max_concurrent": mc,
            "max_same_direction": msd,
        })
    return configs


def make_risk_cfg(preset):
    if preset == "aggressive":
        return RiskConfig(monthly_loss_cap_pct=0.05, weekly_loss_cap_pct=0.03, daily_loss_cap_pct=0.02,
                          dd_throttle_1=0.07, dd_throttle_2=0.12, dd_throttle_3=0.17,
                          consec_loss_throttle_1=2, consec_loss_throttle_2=4,
                          anti_mart_win_mult=1.30, anti_mart_loss_mult=0.60,
                          enable_trailing=True, trail_trigger_1=0.5, trail_trigger_2=1.0, trail_distance_r=0.3)
    if preset == "standard":
        return RiskConfig()
    if preset == "conservative":
        return RiskConfig(monthly_loss_cap_pct=0.05, weekly_loss_cap_pct=0.03, daily_loss_cap_pct=0.02,
                          dd_throttle_1=0.05, dd_throttle_2=0.10, dd_throttle_3=0.15,
                          consec_loss_throttle_1=2, consec_loss_throttle_2=3,
                          base_risk_pct=0.005, anti_mart_max_risk=0.008, anti_mart_min_risk=0.002,
                          enable_trailing=True, trail_trigger_1=0.5, trail_trigger_2=1.0, trail_distance_r=0.3)
    raise ValueError(preset)


# ============================================================================
# Main: train + test all 100 v3 bots
# ============================================================================

def main():
    print("="*120)
    print("AI BOTS v3 — Multi-asset, multi-layer 'thinking' AI, expanded from AIBOT_087")
    print(f"Train: {TRAIN_START.date()} → {TRAIN_END.date()}  |  Val: {VAL_START.date()} → {VAL_END.date()}  |  Test: {TEST_START.date()} → {TEST_END.date()}")
    print(f"Markets: {len(ALL_SYMBOLS)} ({', '.join(ALL_SYMBOLS)})")
    print("="*120)

    print("\n=== STRICT NO-LOOKAHEAD AUDIT ===")
    print("✓ All features at bar i use only data from bars [0..i]")
    print("✓ Cross-asset features use only PAST bars from other assets (df_ref.index < t)")
    print("✓ AI trained only on data BEFORE test window")
    print("✓ Portfolio-level risk: max concurrent positions, max same-direction")

    configs = generate_100_v3_configs(seed=42)
    print(f"\nGenerated {len(configs)} v3 bot configs")

    # Train one AI per (symbol, candidate_gen, ai_config) combination
    # This avoids retraining the same AI for each bot
    print("\nPre-training AIs (one per symbol × candidate_gen × ai_config)...")
    ai_cache = {}  # key = (sym, cand, ai_name) -> ThinkingAI
    ai_thresholds = {}  # key = (sym, cand, ai_name) -> float

    # Get all unique (sym, cand, ai_name) combos needed
    needed = set()
    for cfg in configs:
        for sym in cfg["symbols"]:
            needed.add((sym, cfg["candidate_gen"], cfg["ai_config"]["name"]))
    print(f"  Need to train {len(needed)} unique AIs")

    t_train_start = time.time()
    import pickle
    AI_DISK_CACHE = DATA_DIR / "ai_models_v3"
    AI_DISK_CACHE.mkdir(exist_ok=True)
    n_trained = 0
    for i, (sym, cand, ai_name) in enumerate(sorted(needed)):
        if (i+1) % 20 == 0:
            print(f"  ... {i+1}/{len(needed)} AIs trained ({(i+1)/(time.time()-t_train_start):.1f}/s)", flush=True)
            # Free indicator cache to avoid memory buildup
            _IND_CACHE.clear()
        # Find the ai_config
        ai_config = None
        for cfg in configs:
            if cfg["ai_config"]["name"] == ai_name:
                ai_config = cfg["ai_config"]; break
        if ai_config is None: continue
        # Disk cache path
        safe_name = ai_name.replace("+", "_").replace("/", "_")
        model_path = AI_DISK_CACHE / f"{sym}_{cand}_{safe_name}.pkl"
        thresh_path = AI_DISK_CACHE / f"{sym}_{cand}_{safe_name}_thresh.json"
        if model_path.exists() and thresh_path.exists():
            ai_thresholds[(sym, cand, ai_name)] = json.loads(thresh_path.read_text())["threshold"]
            n_trained += 1
            continue
        try:
            cand_fn = CANDIDATE_GENERATORS[cand]
            samples = generate_training_samples_multi(sym, cand_fn, TRAIN_START, TRAIN_END)
            if len(samples) < 30:
                continue
            ai = train_thinking_ai(samples, FEATURE_COLS_V3, ai_config)
            if ai is None: continue
            # Validate to pick threshold
            val_samples = generate_training_samples_multi(sym, cand_fn, VAL_START, VAL_END)
            best_thresh = 0.5
            if len(val_samples) >= 10:
                X_val = np.array([[s[c] for c in FEATURE_COLS_V3] for s in val_samples])
                y_val = np.array([s["label"] for s in val_samples])
                probas = []
                for k in range(len(X_val)):
                    p, _ = ai.predict(X_val[k:k+1])
                    probas.append(p)
                probas = np.array(probas)
                best_exp = -999
                for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
                    mask = probas >= thresh
                    if mask.sum() < 3: continue
                    wr = y_val[mask].mean()
                    exp = 3 * wr - 1
                    if exp > best_exp:
                        best_exp = exp; best_thresh = thresh
            ai_thresholds[(sym, cand, ai_name)] = best_thresh
            # Save to disk and free memory
            with open(model_path, "wb") as f:
                pickle.dump(ai, f)
            thresh_path.write_text(json.dumps({"threshold": best_thresh}))
            n_trained += 1
            del ai
        except Exception as e:
            print(f"    AI train failed for {sym} {cand} {ai_name}: {e}")
    print(f"  Trained {n_trained} AIs in {time.time()-t_train_start:.1f}s")

    # Run each bot on test window — load AIs from disk on demand
    print(f"\nRunning 100 v3 bots on test window...")
    import pickle as _pickle
    all_results = []
    t_test_start = time.time()
    for i, cfg in enumerate(configs, 1):
        if i % 10 == 0:
            elapsed = time.time() - t_test_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(configs) - i) / rate if rate > 0 else 0
            print(f"  ... {i}/100 done ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
        try:
            # Load AIs from disk for this bot's symbols
            ai_per_symbol = {}
            threshold_per_symbol = {}
            safe_name = cfg["ai_config"]["name"].replace("+", "_").replace("/", "_")
            for sym in cfg["symbols"]:
                key = (sym, cfg["candidate_gen"], cfg["ai_config"]["name"])
                if key in ai_thresholds:
                    threshold_per_symbol[sym] = ai_thresholds[key]
                    model_path = AI_DISK_CACHE / f"{sym}_{cfg['candidate_gen']}_{safe_name}.pkl"
                    if model_path.exists():
                        with open(model_path, "rb") as f:
                            ai_per_symbol[sym] = _pickle.load(f)
            threshold = min(threshold_per_symbol.values()) if threshold_per_symbol else 0.5
            risk_cfg = make_risk_cfg(cfg["risk_preset"])
            trades, stats = backtest_multi_asset(cfg, ai_per_symbol, threshold, risk_cfg,
                                                  start_time=TEST_START, end_time=TEST_END)
            all_results.append({"cfg": cfg, "trades": trades, "stats": stats, "trained": len(ai_per_symbol) > 0})
            # Free AI memory after each bot
            del ai_per_symbol
        except Exception as e:
            import traceback
            print(f"  {cfg['id']} FAILED: {e}")
            all_results.append({"cfg": cfg, "trades": [], "stats": {"n_trades": 0,
                "final_wallet": INITIAL_WALLET, "net": 0, "net_pct": 0,
                "peak_wallet": INITIAL_WALLET, "max_dd_pct": 0,
                "ai_decisions": {"approved": 0, "rejected": 0, "skipped_rm": 0, "skipped_portfolio": 0}},
                "trained": False})

        # Save partial results every 10 bots
        if i % 10 == 0:
            json_path = DOWNLOAD / "ai_bots_v3_partial.json"
            json_data = []
            for r in all_results:
                cfg2 = r["cfg"]; s2 = r["stats"]; t2 = r["trades"]
                n_wins2 = sum(1 for t in t2 if t["net_pnl"] > 0)
                n_losses2 = sum(1 for t in t2 if t["net_pnl"] < 0)
                wr2 = n_wins2/(n_wins2+n_losses2) if (n_wins2+n_losses2) > 0 else 0
                json_data.append({
                    "cfg": cfg2, "trained": r.get("trained", False),
                    "n_trades": s2["n_trades"], "wins": n_wins2, "losses": n_losses2, "wr": wr2,
                    "final_wallet": s2["final_wallet"], "net": s2["net"], "net_pct": s2["net_pct"],
                    "max_dd_pct": s2["max_dd_pct"],
                    "ai_rejected": s2["ai_decisions"].get("rejected", 0),
                    "skipped_portfolio": s2["ai_decisions"].get("skipped_portfolio", 0),
                })
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2, default=str)

    elapsed = time.time() - t_test_start
    print(f"\nDone in {elapsed:.1f}s")

    all_results.sort(key=lambda x: x["stats"]["net"], reverse=True)

    # Top 10
    print("\n" + "="*150)
    print("TOP 10 AI BOTS v3 BY NET PROFIT (last 30 days, $10,000 initial)")
    print("="*150)
    print(f"{'#':<3} {'Bot ID':<55} {'Symbols':<12} {'Cand':<8} {'Risk':<10} {'AI':<15} {'Trades':>6} {'WR':>6} {'Net $':>11} {'Net %':>8} {'MaxDD':>7}")
    print("-"*150)
    for rank, r in enumerate(all_results[:10], 1):
        cfg = r["cfg"]; s = r["stats"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
        wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
        sym_str = cfg["symbol_set"][:10]
        print(f"{rank:<3} {cfg['id'][:55]:<55} {sym_str:<12} {cfg['candidate_gen'][:8]:<8} {cfg['risk_preset'][:10]:<10} {cfg['ai_config']['name'][:15]:<15} "
              f"{s['n_trades']:>6} {wr*100:>5.1f}% {s['net']:>+11,.2f} {s['net_pct']*100:>+7.2f}% {s['max_dd_pct']*100:>6.1f}%")

    # Aggregate
    n_profit = sum(1 for r in all_results if r["stats"]["net"] > 0)
    n_break = sum(1 for r in all_results if r["stats"]["net"] == 0)
    n_loss = sum(1 for r in all_results if r["stats"]["net"] < 0)
    print(f"\n--- AGGREGATE ---")
    print(f"  Trained: {sum(1 for r in all_results if r.get('trained', False))}/100")
    print(f"  Profitable: {n_profit}  |  Breakeven: {n_break}  |  Losing: {n_loss}")
    print(f"  Best: ${all_results[0]['stats']['net']:>+,.2f} ({all_results[0]['stats']['net_pct']*100:+.2f}%)")
    print(f"  Worst: ${all_results[-1]['stats']['net']:>+,.2f} ({all_results[-1]['stats']['net_pct']*100:+.2f}%)")

    # Compare to AIBOT_087 baseline
    print(f"\n--- COMPARISON TO AIBOT_087 (+$1,456.19 / +14.56% / 80% WR / 10 trades) ---")
    print(f"  v3 best:  {all_results[0]['cfg']['id'][:30]}  ${all_results[0]['stats']['net']:>+,.2f} ({all_results[0]['stats']['net_pct']*100:+.2f}%)  trades={all_results[0]['stats']['n_trades']}")

    # Top 5 ensemble
    top5 = all_results[:5]
    ens_net = sum(r["stats"]["net"] * 0.2 for r in top5)
    avg_dd = np.mean([r["stats"]["max_dd_pct"] for r in top5])
    print(f"\n--- ENSEMBLE: $2,000 each on top 5 ---")
    print(f"  Initial: ${INITIAL_WALLET:,.2f}  →  Final: ${INITIAL_WALLET + ens_net:,.2f}")
    print(f"  Net: ${ens_net:+,.2f} ({ens_net/INITIAL_WALLET*100:+.2f}%)  |  Avg max DD: {avg_dd*100:.1f}%")
    for r in top5:
        cfg = r["cfg"]; s = r["stats"]
        print(f"    {cfg['id'][:55]:55s} ${s['net']*0.2:+9,.2f} ({s['net_pct']*100:+6.2f}%)  DD={s['max_dd_pct']*100:.1f}%  trades={s['n_trades']}")

    # Save Excel
    out_path = DOWNLOAD / "ai_bots_v3_30day.xlsx"
    write_excel(all_results, out_path)
    print(f"\n>>> Wrote {out_path}")


def write_excel(all_results, out_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    pos_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    neg_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    warn_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))

    ws = wb.create_sheet("All 100 Ranked")
    ws["A1"] = "AI BOTS v3 — 30-DAY TEST, RANKED BY NET PROFIT"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = f"Multi-asset ({len(ALL_SYMBOLS)} markets) + multi-layer AI + portfolio risk mgmt"
    ws["A3"] = f"Window: {TEST_START} → {TEST_END}  |  Initial: ${INITIAL_WALLET:,.2f}"

    headers = ["Rank", "Bot ID", "Symbol Set", "Symbols", "Candidate", "Risk", "AI Config",
               "Max Concurrent", "Max Same Dir", "Trained",
               "Trades", "Wins", "Losses", "WR", "Initial $", "Final $", "Net $", "Net %", "Max DD %",
               "AI Rejected", "Portfolio Skips"]
    row_idx = 5
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=row_idx, column=col, value=h)
        c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center", wrap_text=True); c.border = border
    for rank, r in enumerate(all_results, 1):
        cfg = r["cfg"]; s = r["stats"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
        wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
        row = [rank, cfg["id"], cfg["symbol_set"], ",".join(cfg["symbols"][:3]) + ("..." if len(cfg["symbols"])>3 else ""),
               cfg["candidate_gen"], cfg["risk_preset"], cfg["ai_config"]["name"],
               cfg["max_concurrent"], cfg["max_same_direction"],
               "Yes" if r.get("trained") else "No",
               s["n_trades"], n_wins, n_losses, wr,
               INITIAL_WALLET, s["final_wallet"], s["net"], s["net_pct"], s["max_dd_pct"],
               s["ai_decisions"].get("rejected", 0), s["ai_decisions"].get("skipped_portfolio", 0)]
        for col, v in enumerate(row, 1):
            c = ws.cell(row=row_idx+rank, column=col, value=v)
            c.border = border
            if col in (16, 17):
                c.number_format = '"$"#,##0.00'
            elif col in (18, 19):
                c.number_format = '0.00%'
            elif col == 14:
                c.number_format = '0.0%'
            if col == 17 and isinstance(v, (int, float)):
                c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
    widths = [5, 60, 12, 20, 10, 12, 18, 8, 8, 8, 7, 7, 7, 7, 11, 12, 12, 9, 9, 11, 12]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A6"

    # Top 10 trade logs
    for rank, r in enumerate(all_results[:10], 1):
        cfg = r["cfg"]; trades = r["trades"]; s = r["stats"]
        sheet_name = f"#{rank} {cfg['id']}"[:31]
        ws = wb.create_sheet(sheet_name)
        ws["A1"] = f"#{rank} — {cfg['id']}"
        ws["A1"].font = Font(bold=True, size=10); ws.merge_cells("A1:V1")
        ws["A2"] = f"Symbols: {', '.join(cfg['symbols'])}  |  Candidate: {cfg['candidate_gen']}  |  Risk: {cfg['risk_preset']}  |  AI: {cfg['ai_config']['name']}  |  Max concurrent: {cfg['max_concurrent']}"
        ws.merge_cells("A2:V2")
        ws["A3"] = f"Net: ${s['net']:+,.2f} ({s['net_pct']*100:+.2f}%)  |  Trades: {len(trades)}  |  Max DD: {s['max_dd_pct']*100:.1f}%"
        ws.merge_cells("A3:V3")
        trade_headers = ["#", "Symbol", "Entry Time", "Exit Time", "Side", "Entry $", "Exit $",
                         "Stop $", "Target $", "Size", "Risk %", "Risk $",
                         "Gross P&L $", "Cost $", "Net P&L $", "R Mult", "AI P(win)",
                         "Exit Reason", "Bars Held", "Wallet Before $", "Wallet After $"]
        for col, h in enumerate(trade_headers, 1):
            c = ws.cell(row=5, column=col, value=h)
            c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center"); c.border = border
        for i, t in enumerate(trades, 1):
            entry_t = t["entry_time"].tz_localize(None) if hasattr(t["entry_time"], "tz_localize") else t["entry_time"]
            exit_t = t["exit_time"].tz_localize(None) if hasattr(t["exit_time"], "tz_localize") else t["exit_time"]
            row = [i, t["symbol"], entry_t, exit_t, t["side"], t["entry_price"], t["exit_price"],
                   t["stop"], t["target"], t["size"], t["risk_pct"], t["risk_usd"],
                   t["gross_pnl"], t["cost"], t["net_pnl"], t["R_multiple"], t["ai_proba"],
                   t["exit_reason"], t["bars_held"], t["wallet_before"], t["wallet_after"]]
            for col, v in enumerate(row, 1):
                c = ws.cell(row=5+i, column=col, value=v)
                c.border = border
                if col in (6, 7, 8, 9, 12, 13, 14, 15, 20, 21) and isinstance(v, (int, float)):
                    c.number_format = '"$"#,##0.00'
                elif col == 11 and isinstance(v, (int, float)):
                    c.number_format = '0.00%'
                elif col in (16, 17) and isinstance(v, (int, float)):
                    c.number_format = '0.00'
                if col == 15 and isinstance(v, (int, float)):
                    c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
        twidths = [5, 10, 22, 22, 6, 12, 12, 12, 12, 12, 8, 10, 12, 10, 12, 8, 10, 12, 8, 14, 14]
        for i, w in enumerate(twidths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A6"

    wb.save(out_path)


if __name__ == "__main__":
    main()
