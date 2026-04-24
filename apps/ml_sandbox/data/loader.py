"""
loader.py
─────────
Carga datos para entrenamiento y test según la configuración del yaml.

Si el modelo es sklearn → carga desde silver_features (Supabase)
Si el modelo es pytorch → carga desde tensores locales (.npy)

Cambios: usa shared.db.sb (singleton) en vez de create_client.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from shared.db import sb
from apps.ml_sandbox.config import ExperimentConfig

log = logging.getLogger(__name__)


def load_data(cfg: ExperimentConfig) -> pd.DataFrame:
    """Carga datos según el tipo de modelo definido en el yaml."""
    if cfg.is_sklearn:
        return _load_from_silver(cfg)
    else:
        return _load_from_tensor(cfg)


def _load_from_silver(cfg: ExperimentConfig) -> pd.DataFrame:
    """Carga datos desde silver_features en Supabase."""
    all_dfs = []

    select_cols = ["ts", "ticker"] + cfg.data.columns + [cfg.data.target]
    select_cols = list(dict.fromkeys(select_cols))

    for table in cfg.data.tables:
        log.info(f"Cargando {table}...")
        for ticker in cfg.data.tickers:
            rows: list[dict] = []
            offset = 0

            while True:
                try:
                    resp = (
                        sb.table(table)
                        .select(",".join(select_cols))
                        .eq("ticker", ticker)
                        .gte("ts", cfg.data.train_start)
                        .lte("ts", cfg.data.test_end)
                        .order("ts")
                        .range(offset, offset + 999)
                        .execute()
                    )
                    batch = resp.data or []
                    rows.extend(batch)
                    if len(batch) < 1000:
                        break
                    offset += 1000
                except Exception as e:
                    log.error(f"Error cargando {ticker} de {table}: {e}")
                    break

            if rows:
                df = pd.DataFrame(rows)
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                all_dfs.append(df)
                log.info(f"  {ticker}: {len(df)} filas")

    if not all_dfs:
        raise ValueError("No se encontraron datos en Supabase para los parámetros dados")

    result = pd.concat(all_dfs).sort_values(["ts", "ticker"])

    if cfg.data.dropna:
        before = len(result)
        result = result.dropna(subset=cfg.data.columns + [cfg.data.target])
        log.info(f"Filas eliminadas por NaN: {before - len(result)}")

    log.info(f"Total filas cargadas: {len(result)}")
    return result


def _load_from_tensor(cfg: ExperimentConfig) -> pd.DataFrame:
    """Carga datos desde tensores locales .npy."""
    tensor_path = (
        cfg.tensors_dir / cfg.data.tensor_interval
        / f"tensor_{cfg.data.tensor_type}.npy"
    )
    meta_path = (
        cfg.tensors_dir / cfg.data.tensor_interval
        / f"tensor_{cfg.data.tensor_type}_meta.npz"
    )

    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor no encontrado: {tensor_path}")

    log.info(f"Cargando tensor: {tensor_path}")
    tensor = np.load(tensor_path)
    meta = np.load(meta_path, allow_pickle=True)

    timestamps = pd.to_datetime(meta["timestamps"])
    tickers = list(meta["tickers"])
    column_names = (
        [str(c) for c in meta["columns"]]
        if "columns" in meta
        else [f"feature_{i}" for i in range(tensor.shape[2])]
    )

    ticker_indices = [tickers.index(t) for t in cfg.data.tickers if t in tickers]
    if not ticker_indices:
        raise ValueError(
            f"Ningún ticker del yaml encontrado en el tensor: {cfg.data.tickers}"
        )

    ts_mask = (
        (timestamps >= pd.Timestamp(cfg.data.train_start, tz="UTC"))
        & (timestamps <= pd.Timestamp(cfg.data.test_end, tz="UTC"))
    )
    tensor_filtered = tensor[ts_mask]
    ts_filtered = timestamps[ts_mask]

    rows = []
    for t_idx, ticker in zip(ticker_indices, cfg.data.tickers):
        for i, ts in enumerate(ts_filtered):
            row = {"ts": ts, "ticker": ticker}
            for f_idx, col_name in enumerate(column_names):
                row[col_name] = tensor_filtered[i, t_idx, f_idx]
            rows.append(row)

    result = pd.DataFrame(rows)
    log.info(f"Tensor cargado: {result.shape} — columnas: {column_names[:5]}...")
    return result
