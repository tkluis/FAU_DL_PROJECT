from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

class FlatWindowData(Data):
    """PyG Data subclass for flattened node-time tensors.

    The original STGAT code stores `x` and `r` as flattened tensors of shape
    (num_nodes * window, 1). PyG batching would otherwise infer num_nodes from
    x.size(0), which is incorrect for graph connectivity (edge_index expects num_nodes).

    This class forces edge_index increments to use `self.num_nodes`.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            # Ensure batching offsets edges by the true number of graph nodes.
            if self.num_nodes is None:
                raise RuntimeError("FlatWindowData.num_nodes must be set for batching.")
            return int(self.num_nodes)
        return super().__inc__(key, value, *args, **kwargs)


try:
    import yfinance as yf
except Exception as e:
    raise ImportError(
        "yfinance is required. Install with: pip install yfinance"
    ) from e


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_yahoo_ticker(t: str) -> str:
    """Normalize common ticker quirks for Yahoo Finance."""
    t = t.strip()
    # Yahoo uses '-' for class shares (BRK.B -> BRK-B, BF.B -> BF-B)
    t = t.replace(".", "-")
    return t


def read_tickers(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = [x.strip() for x in f.read().splitlines() if x.strip() and not x.strip().startswith("#")]
    tickers = [normalize_yahoo_ticker(x) for x in raw]

    # de-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def safe_panel(panel: pd.DataFrame, min_coverage: float = 0.95) -> pd.DataFrame:
    """Forward-fill and drop dates/tickers with excessive missingness."""
    panel = panel.sort_index()
    panel = panel.ffill()

    # Drop rows with too many NaNs
    thresh = int(panel.shape[1] * min_coverage)
    panel = panel.dropna(axis=0, thresh=thresh)

    # Drop any remaining columns with NaN
    panel = panel.dropna(axis=1, how="any")
    return panel


def download_or_load_yf(
    tickers: List[str],
    start: str,
    end: str,
    cache_path: str,
    refresh: bool = False
) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance or load from a local parquet cache."""
    ensure_dir(os.path.dirname(cache_path))

    if (not refresh) and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=False,
        progress=False,
    )

    # Normalize to columns = (ticker, field)
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.levels[0].tolist()
        if "Open" in level0 or "Close" in level0:
            df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    df.to_parquet(cache_path)
    return df


def extract_panel(df: pd.DataFrame, field: str, tickers: List[str]) -> pd.DataFrame:
    field_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }
    key = field_map[field.lower()]
    cols = [(t, key) for t in tickers if (t, key) in df.columns]
    panel = df[cols].copy()
    panel.columns = [c[0] for c in panel.columns]
    return panel


def compute_features(close: pd.DataFrame, volume: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Feature panels (date x ticker)."""
    log_close = np.log(close)
    ret = log_close.diff()

    vol5 = ret.rolling(5).std()
    mom10 = ret.rolling(10).mean()

    vol_clean = volume.replace(0, np.nan)
    dlogvol = np.log(vol_clean).diff()

    return {"ret": ret, "vol5": vol5, "mom10": mom10, "dlogvol": dlogvol}


def build_correlation_graph(returns: np.ndarray, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Static correlation graph -> (edge_index, edge_weight)."""
    corr = np.corrcoef(returns.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    src, dst, w = [], [], []
    n = corr.shape[0]
    for i in range(n):
        nbrs = np.argsort(-np.abs(corr[i]))[:top_k]
        for j in nbrs:
            src.append(i); dst.append(j); w.append(float(corr[i, j]))

    # undirected
    src2 = src + dst
    dst2 = dst + src
    w2 = w + w

    edge_index = torch.tensor([src2, dst2], dtype=torch.long)
    edge_weight = torch.tensor(w2, dtype=torch.float32).view(-1, 1)
    return edge_index, edge_weight


@dataclass
class YFConfig:
    start: str = "2018-01-01"
    end: str = "2024-12-31"
    window: int = 20
    horizon: int = 1
    top_k: int = 10
    seed: int = 42
    cache_dir: str = "data_cache"
    tickers_file: str = "tickers.txt"
    refresh: bool = False

    # Index proxies for portfolio scripts
    idx_sp500: str = "SPY"
    idx_djia: str = "DIA"
    idx_nasdaq: str = "QQQ"
    idx_csi_proxy: str = "FXI"  # proxy ETF for China large-cap (optional)


def gen_GNN_data(cfg: Optional[YFConfig] = None) -> Tuple[List[Data], float, float, float]:
    """Generate a list of PyG Data graphs from Yahoo Finance OHLCV.

    Returns:
        Gdata_list: list[Data]
        split: float (compatibility output)
        max_value, min_value: float (compatibility output)

    Notes:
    - The STGAT model in this repo reshapes tensors assuming a fixed window=20.
      This generator uses cfg.window=20 by default to match that behavior.
    - Targets are next-day log returns per ticker (shape: [num_nodes]).
    - Portfolio scripts also expect additional fields (gourujia, std_dev_next, etc.).
    """
    if cfg is None:
        cfg = YFConfig()

    fix_seed(cfg.seed)

    if not os.path.exists(cfg.tickers_file):
        raise FileNotFoundError(
            f"Missing {cfg.tickers_file}. Create it with one Yahoo ticker per line.\n"
            "Example:\nAAPL\nMSFT\nAMZN\nGOOGL\nMETA\nNVDA\nTSLA\nJPM\nXOM\nUNH\n"
        )

    stock_tickers = read_tickers(cfg.tickers_file)
    idx_tickers = [
        normalize_yahoo_ticker(cfg.idx_sp500),
        normalize_yahoo_ticker(cfg.idx_djia),
        normalize_yahoo_ticker(cfg.idx_nasdaq),
        normalize_yahoo_ticker(cfg.idx_csi_proxy),
    ]
    all_tickers = stock_tickers + [t for t in idx_tickers if t not in stock_tickers]

    cache_path = os.path.join(cfg.cache_dir, f"yf_{cfg.start}_{cfg.end}.parquet")
    df = download_or_load_yf(all_tickers, cfg.start, cfg.end, cache_path, refresh=cfg.refresh)

    close_all = extract_panel(df, "close", all_tickers)
    open_all = extract_panel(df, "open", all_tickers)
    vol_all = extract_panel(df, "volume", all_tickers)

    close_all = safe_panel(close_all, min_coverage=0.95)
    open_all = open_all.reindex(close_all.index).ffill()
    vol_all = vol_all.reindex(close_all.index).ffill()

    # Stock-only panels
    close = close_all[stock_tickers].copy()
    open_ = open_all[stock_tickers].copy()
    vol = vol_all[stock_tickers].copy()

    # Index series (used by portfolio scripts)
    def idx_series(ticker: str) -> pd.Series:
        if ticker in close_all.columns:
            return close_all[ticker].copy()
        return pd.Series(index=close_all.index, data=np.nan).ffill()

    sp500 = idx_series(idx_tickers[0])
    djia = idx_series(idx_tickers[1])
    nasdaq = idx_series(idx_tickers[2])
    csi = idx_series(idx_tickers[3])

    feats = compute_features(close, vol)

    # Align features (drop initial NaNs due to rolling)
    feat_panel = pd.concat(feats.values(), axis=1, keys=list(feats.keys())).dropna()

    # Target: next-day return (log-return)
    ret = feats["ret"].reindex(feat_panel.index)
    y_target = ret.shift(-cfg.horizon).reindex(feat_panel.index).dropna()
    feat_panel = feat_panel.reindex(y_target.index)

    # Graph from returns
    returns_np = feats["ret"].reindex(feat_panel.index).dropna().values
    edge_index, edge_weight = build_correlation_graph(returns_np, top_k=cfg.top_k)

    # Compatibility outputs
    split = 0.8
    max_value = float(np.nanmax(close.values))
    min_value = float(np.nanmin(close.values))

    # Helper: index daily log return
    def idx_ret(series: pd.Series, day: pd.Timestamp) -> float:
        r = np.log(series).diff()
        val = r.loc[day] if day in r.index else np.nan
        if pd.isna(val) or np.isinf(val):
            return 0.0
        return float(val)

    # Additional per-sample arrays
    close_change = np.log(close).diff().fillna(0.0)

    dates = list(feat_panel.index)
    Gdata_list: List[Data] = []
    num_nodes = len(stock_tickers)

    for i in range(cfg.window - 1, len(dates) - cfg.horizon):
        day = dates[i]
        w_idx = dates[i - cfg.window + 1 : i + 1]

        # t and s are shaped (N, window)
        t_mat = feats["ret"].reindex(w_idx).values
        t_mat = np.nan_to_num(t_mat, nan=0.0, posinf=0.0, neginf=0.0).T.astype(np.float32)

        s_mat = close_change.reindex(w_idx).values
        s_mat = np.nan_to_num(s_mat, nan=0.0, posinf=0.0, neginf=0.0).T.astype(np.float32)

        # x and r are flattened (N*window, 1)
        x_flat = t_mat.T.reshape(num_nodes * cfg.window, 1).astype(np.float32)
        r_mat = (t_mat - t_mat.mean(axis=1, keepdims=True)).astype(np.float32)
        r_flat = r_mat.T.reshape(num_nodes * cfg.window, 1).astype(np.float32)

        # labels and portfolio helpers
        y_vec = y_target.loc[day].values.astype(np.float32)  # (N,)
        buy_price = close.loc[day].values.astype(np.float32)  # (N,)
        std_next = feats["ret"].reindex(w_idx).std(axis=0).values.astype(np.float32)  # (N,)
        zhangting = np.zeros((num_nodes,), dtype=np.float32)  # not applicable; keep for compatibility

        data = FlatWindowData(num_nodes=num_nodes, 
            x=torch.from_numpy(x_flat),
            r=torch.from_numpy(r_flat),
            t=torch.from_numpy(t_mat),
            s=torch.from_numpy(s_mat),
            edge_index=edge_index,
            edge_weight=edge_weight,
            shouchujia=torch.from_numpy(y_vec),
            gourujia=torch.from_numpy(buy_price),
            std_dev_next=torch.from_numpy(std_next),
            zhangting=torch.from_numpy(zhangting),
            riqi=str(day.date()),
            biaopuzhishu=torch.tensor(idx_ret(sp500, day), dtype=torch.float32),
            DJIA_zhishu=torch.tensor(idx_ret(djia, day), dtype=torch.float32),
            Nasdaq_zhishu=torch.tensor(idx_ret(nasdaq, day), dtype=torch.float32),
            zzzs=torch.tensor(idx_ret(csi, day), dtype=torch.float32),
        )
        data.num_nodes = num_nodes
        Gdata_list.append(data)

    return Gdata_list, split, max_value, min_value
