"""
indicators.py
─────────────
Indicadores técnicos unificados. Esta es la FUENTE ÚNICA de verdad
para el cálculo de features, usada tanto por el pipeline histórico
(apps/ingestion_historical/silver.py) como por el live
(apps/ingestion_live/silver_rt.py).

Resuelve los issues F-88, F-89, F-90, F-91 del informe de Fase 2
(divergencia de fórmulas entre training y live).

Fórmulas usadas (las "canónicas" del silver.py original):
    - RSI: EMA Wilder (alpha=1/period)
    - ATR: EMA Wilder
    - MACD: EMA 12/26/9 estándar
    - Bollinger: SMA 20 + 2 std
    - VWAP: reset diario (groupby por fecha)
    - Returns: log returns np.log(close/shift(1))

TODO en Fase 3.4: implementar estas funciones y migrar
silver.py y silver_rt.py para que las consuman.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average estándar (adjust=False)."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder (EMA-based). Estándar de la literatura técnica.
    
    IMPORTANTE: silver_rt.py actual usa SMA en lugar de EMA aquí.
    Esta es la versión canónica.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD estándar (12, 26, 9). Retorna (line, signal, histogram)."""
    ema_fast = ema(close, 12)
    ema_slow = ema(close, 26)
    macd_line = ema_fast - ema_slow
    macd_sig = ema(macd_line, 9)
    macd_hist = macd_line - macd_sig
    return macd_line, macd_sig, macd_hist


def bollinger(
    close: pd.Series, period: int = 20, std_multiplier: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (period=20, std=2). Retorna (upper, middle, lower, width, pct_b)."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_multiplier * std
    lower = middle - std_multiplier * std
    width = (upper - lower) / middle.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, middle, lower, width, pct_b


def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP con reset diario (groupby por fecha).
    
    IMPORTANTE: silver_rt.py actual usa cumsum() sin reset.
    Esta es la versión canónica (reset diario cada sesión).
    
    df debe tener columnas: high, low, close, volume, y un índice datetime.
    """
    df = df.copy()
    df["_date"] = df.index.date if hasattr(df.index, "date") else pd.to_datetime(df["ts"]).dt.date
    df["_tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpvol"] = df["_tp"] * df["volume"]
    cum_tpvol = df.groupby("_date")["_tpvol"].cumsum()
    cum_vol = df.groupby("_date")["volume"].cumsum()
    return cum_tpvol / cum_vol.replace(0, np.nan)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range Wilder (EMA-based).
    
    IMPORTANTE: silver_rt.py actual usa SMA en lugar de EMA.
    Esta es la versión canónica.
    
    df debe tener columnas: high, low, close.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Log returns. np.log(close / shift(period)).
    
    IMPORTANTE: silver_rt.py actual usa pct_change() (simple returns).
    Esta es la versión canónica (log returns).
    """
    return np.log(close / close.shift(period))


# TODO Fase 3.4:
# - Añadir is_market_open con DST + festivos (resolverá F-76, F-77, F-92)
# - Añadir sentiment_at_timestamp con ventana 15min (resolverá F-93)
# - Tests unitarios comparando training vs live
