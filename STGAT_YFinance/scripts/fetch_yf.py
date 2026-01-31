"""scripts/fetch_yf.py

Standalone Yahoo Finance downloader + cache builder.

Why this exists:
- yfinance may use internal caching that can hit SQLite locking when downloads are threaded.
- This script defaults to **threads=False** and retries failed tickers one-by-one.

Usage:
  python scripts/fetch_yf.py --tickers tickers.txt --start 2018-01-01 --end 2024-12-31

Outputs:
  data_cache/yf_<start>_<end>.parquet
"""

import argparse
import os
import time
import pandas as pd
import yfinance as yf

def normalize(t: str) -> str:
    return t.strip().replace(".", "-")

def read_tickers(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [normalize(x) for x in f.read().splitlines() if x.strip() and not x.strip().startswith("#")]

def download_batch(tickers, start, end):
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=False,   # IMPORTANT: avoid sqlite locking in some environments
        progress=True,
    )

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.levels[0].tolist()
        if "Open" in level0 or "Close" in level0:
            df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="tickers.txt")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default=None)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    tickers = read_tickers(args.tickers)
    if not tickers:
        raise ValueError("No tickers found in tickers file.")

    out = args.out or os.path.join("data_cache", f"yf_{args.start}_{args.end}.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # First pass: one batch (threads disabled)
    df = download_batch(tickers, args.start, args.end)
    df = normalize_columns(df)

    # Detect missing tickers by missing Close column
    missing = []
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") not in df.columns:
                missing.append(t)
    else:
        # Single ticker case
        if "Close" not in df.columns:
            missing = tickers[:]  # treat as missing

    # Retry missing tickers individually and merge
    for t in missing:
        ok = False
        for attempt in range(1, args.retries + 1):
            try:
                print(f"Retry {attempt}/{args.retries} for {t} ...")
                dfi = download_batch([t], args.start, args.end)
                dfi = normalize_columns(dfi)

                if isinstance(dfi.columns, pd.MultiIndex):
                    df = df.join(dfi, how="outer")
                else:
                    # convert flat cols to MultiIndex (ticker, field)
                    dfi.columns = pd.MultiIndex.from_product([[t], dfi.columns])
                    if not isinstance(df.columns, pd.MultiIndex):
                        df.columns = pd.MultiIndex.from_product([[tickers[0]], df.columns])
                    df = df.join(dfi, how="outer")

                ok = True
                time.sleep(0.5)
                break
            except Exception as e:
                print(f"  failed for {t}: {e}")
                time.sleep(1.0)

        if not ok:
            print(f"WARNING: Could not download {t} after retries; consider removing it from tickers.txt")

    df.to_parquet(out)
    print("Saved:", out, "shape:", df.shape)

if __name__ == "__main__":
    main()
