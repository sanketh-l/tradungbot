"""
Phase 1: Data acquisition and caching.

Downloads BTCUSDT and ETHUSDT OHLCV at 15-minute and 1-hour timeframes from
Binance public REST API, caches as Parquet, and logs metadata.

Caching is mandatory: subsequent runs read from ./data/ and never re-download
unless --force is passed.

Binance klines endpoint: https://api.binance.com/api/v3/klines
  - Max 1000 candles per request.
  - Public, no API key required.
  - Rate limit: 1200 weight/min; each klines call costs weight 1 (compact) to 2.
  - We throttle to be safe.

Source spec: spot klines, UTC, open_time in ms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Symbols and timeframes to fetch.
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["15m", "1h"]

# We want 5+ years. Binance launched BTCUSDT spot in 2017, ETHUSDT in 2017.
# Use 2019-01-01 as start (well after launch, deep liquidity) to today.
DEFAULT_START = "2019-01-01"
DEFAULT_END = None  # None = now

# Columns returned by Binance klines (we keep the ones we need).
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_klines_paginated(symbol: str, interval: str, start_ms: int, end_ms: int,
                           session: requests.Session) -> list[list]:
    """Page through Binance klines 1000 at a time. Returns list of raw rows."""
    all_rows: list[list] = []
    cur = start_ms
    page = 0
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        for attempt in range(5):
            try:
                r = session.get(BASE_URL, params=params, timeout=30)
                if r.status_code == 429 or r.status_code == 418:
                    # Rate limited. Back off.
                    wait = 2 ** (attempt + 2)
                    print(f"  rate-limited ({r.status_code}); sleeping {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"  attempt {attempt+1} failed: {e}; sleeping {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to fetch {symbol} {interval} after retries")

        rows = r.json()
        if not rows:
            break

        all_rows.extend(rows)
        last_close_time = rows[-1][6]  # close_time in ms
        # Advance cursor past last close time
        cur = last_close_time + 1
        page += 1
        if page % 20 == 0:
            print(f"  ... {symbol} {interval}: {len(all_rows)} candles so far")
        # Binance rate limit safety: small sleep per call
        time.sleep(0.15)
        if len(rows) < 1000:
            break
    return all_rows


def build_dataframe(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df = df.drop(columns=["ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c]).astype("float64")
    df["trades"] = df["trades"].astype("int64")
    # De-duplicate on open_time (defensive; Binance is clean but be safe)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def check_gaps(df: pd.DataFrame, interval: str) -> dict:
    """Report gaps larger than 1 interval. Binance spot rarely has gaps."""
    deltas = df["open_time"].diff().dropna()
    expected = pd.Timedelta(interval) if interval != "1h" else pd.Timedelta(hours=1)
    if interval == "15m":
        expected = pd.Timedelta(minutes=15)
    gaps = deltas[deltas > expected]
    return {
        "expected_interval": str(expected),
        "n_gaps": int(len(gaps)),
        "max_gap": str(deltas.max()) if len(deltas) else None,
        "gap_starts": [str(t) for t in df["open_time"].iloc[gaps.index[:5]].tolist()],
    }


def fetch_one(symbol: str, interval: str, start: str, end: str | None,
              force: bool, session: requests.Session) -> pd.DataFrame:
    cache_path = DATA_DIR / f"{symbol}_{interval}.parquet"
    meta_path = DATA_DIR / f"{symbol}_{interval}.meta.json"
    if cache_path.exists() and not force:
        df = pd.read_parquet(cache_path)
        print(f"  [cache] {symbol} {interval}: {len(df)} rows from {df['open_time'].min()} to {df['open_time'].max()}")
        return df

    start_dt = pd.Timestamp(start, tz="UTC")
    start_ms = int(start_dt.timestamp() * 1000)
    if end is None:
        end_dt = pd.Timestamp.now(tz="UTC")
    else:
        end_dt = pd.Timestamp(end, tz="UTC")
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"  Fetching {symbol} {interval} from {start_dt} to {end_dt}")
    rows = fetch_klines_paginated(symbol, interval, start_ms, end_ms, session)
    if not rows:
        raise RuntimeError(f"No rows returned for {symbol} {interval}")
    df = build_dataframe(rows)
    df.to_parquet(cache_path, index=False)

    gaps = check_gaps(df, interval)
    meta = {
        "symbol": symbol,
        "interval": interval,
        "source_url": BASE_URL,
        "start": str(df["open_time"].min()),
        "end": str(df["open_time"].max()),
        "row_count": int(len(df)),
        "gaps": gaps,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {"start_request": start, "end_request": end or "now"},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  [saved] {cache_path} | {len(df)} rows | gaps={gaps['n_gaps']}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--force", action="store_true", help="Re-download even if cache exists")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "strategy-research-agent/1.0"})

    print(f"=== Phase 1: Data acquisition (start={args.start}, end={args.end or 'now'}) ===")
    summary = []
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"\n-- {sym} {tf} --")
            df = fetch_one(sym, tf, args.start, args.end, args.force, session)
            summary.append({
                "symbol": sym, "tf": tf,
                "rows": len(df),
                "start": str(df["open_time"].min()),
                "end": str(df["open_time"].max()),
            })

    print("\n=== Phase 1 summary ===")
    for s in summary:
        print(f"  {s['symbol']} {s['tf']}: {s['rows']:>7} rows | {s['start']} -> {s['end']}")


if __name__ == "__main__":
    main()
