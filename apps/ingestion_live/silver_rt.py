"""
silver_rt.py
────────────
Calcula indicadores técnicos sobre las barras RT
y los guarda en silver_features_rt en Supabase.

CAMBIO CRÍTICO respecto al original:
    Todos los indicadores ahora vienen de shared.indicators,
    que es la MISMA fuente que usa el pipeline histórico.
    Esto resuelve la divergencia de paridad training/live:

    - F-88: RSI ahora usa EMA Wilder (antes: SMA)
    - F-89: VWAP ahora resetea por día (antes: cumsum sin reset)
    - F-90: ATR ahora usa EMA Wilder (antes: SMA)
    - F-91: Returns ahora usa log returns (antes: pct_change)

Uso:
    from apps.ingestion_live.silver_rt import compute_silver_rt
    df = compute_silver_rt(tickers=["AAPL"], timeframe="1m")
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from shared.db import sb
from shared.indicators import (
    atr,
    bollinger,
    ema,
    log_returns,
    macd,
    rsi,
    vwap,
)

log = logging.getLogger(__name__)


def _is_market_open(ts: pd.Series) -> pd.Series:
    """Detecta si el timestamp está dentro del horario NYSE (13:30-20:00 UTC).

    TODO F-76: añadir soporte DST (invierno es 14:30-21:00 UTC).
    TODO F-77: considerar festivos US con exchange_calendars.
    """
    minutes_since_midnight = ts.dt.hour * 60 + ts.dt.minute
    return ((minutes_since_midnight >= 810) & (minutes_since_midnight < 1200)).astype(int)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos usando shared.indicators (fuente canónica).

    El DataFrame debe tener columnas: ts, ticker, open, high, low, close, volume.
    """
    df = df.sort_values("ts").copy()

    close = df["close"]
    volume = df["volume"]

    # EMAs
    df["ema_9"] = ema(close, 9)
    df["ema_12"] = ema(close, 12)
    df["ema_21"] = ema(close, 21)
    df["ema_50"] = ema(close, 50)

    # RSI — F-88: ahora EMA Wilder (canónico), antes era SMA
    df["rsi_14"] = rsi(close, 14)

    # MACD
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(close)

    # Bollinger Bands
    df["bb_upper"], df["bb_middle"], df["bb_lower"], df["bb_width"], df["bb_pct"] = (
        bollinger(close, 20)
    )

    # VWAP — F-89: ahora reset diario (canónico), antes era cumsum sin reset
    df["vwap"] = vwap(df)

    # ATR — F-90: ahora EMA Wilder (canónico), antes era SMA
    df["atr_14"] = atr(df, 14)

    # Returns — F-91: ahora log returns (canónico), antes era pct_change
    df["returns"] = log_returns(close, 1)
    df["returns_5"] = log_returns(close, 5)
    df["returns_15"] = log_returns(close, 15)
    df["range_pct"] = (df["high"] - df["low"]) / close.replace(0, np.nan)

    # Volume norm
    vol_mean = volume.rolling(20).mean().replace(0, np.nan)
    df["volume_norm"] = volume / vol_mean

    # Time features
    df["hour"] = df["ts"].dt.hour
    df["dayofweek"] = df["ts"].dt.dayofweek

    # Market open
    df["is_market_open"] = _is_market_open(df["ts"])

    return df


def compute_silver_rt(
    tickers: list[str],
    timeframe: str = "1m",
    sentiment: dict | None = None,
) -> pd.DataFrame:
    """
    Lee raw_ohlcv_rt, calcula indicadores y guarda en silver_features_rt.

    Args:
        tickers:   lista de tickers
        timeframe: granularidad
        sentiment: dict con sentiment por ticker {ticker: {label, score, encoded}}

    Returns:
        DataFrame con features calculadas
    """
    all_dfs = []

    for ticker in tickers:
        try:
            # Leer raw_ohlcv_rt
            resp = (
                sb.table("raw_ohlcv_rt")
                .select("ticker,timeframe,ts,open,high,low,close,volume")
                .eq("ticker", ticker)
                .eq("timeframe", timeframe)
                .order("ts", desc=False)
                .limit(200)
                .execute()
            )

            if not resp.data:
                log.warning(f"  {ticker}: sin datos en raw_ohlcv_rt")
                continue

            df = pd.DataFrame(resp.data)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").reset_index(drop=True)

            # Calcular indicadores (ahora con shared.indicators)
            df = compute_indicators(df)

            # Añadir sentiment
            sent = (sentiment or {}).get(ticker, {})
            df["sentiment_label"] = sent.get("label", "neutral")
            df["sentiment_score"] = sent.get("score", 0.0)
            df["sentiment_label_encoded"] = sent.get("encoded", 0)

            # Solo guardar las últimas 100 filas
            df = df.tail(100).copy()

            # Columnas a guardar
            cols = [
                "ts", "ticker", "open", "high", "low", "close", "volume",
                "ema_9", "ema_12", "ema_21", "ema_50",
                "rsi_14", "macd_line", "macd_signal", "macd_hist",
                "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
                "vwap", "atr_14",
                "returns", "returns_5", "returns_15", "range_pct", "volume_norm",
                "hour", "dayofweek", "is_market_open",
                "sentiment_label", "sentiment_score", "sentiment_label_encoded",
            ]

            rows = df[cols].copy()
            rows["ts"] = rows["ts"].astype(str)
            rows = rows.replace({np.nan: None, np.inf: None, -np.inf: None})

            records = rows.to_dict(orient="records")
            if records:
                sb.table("silver_features_rt").upsert(
                    records, on_conflict="ticker,ts"
                ).execute()
                log.info(
                    f"  {ticker}: {len(records)} filas guardadas en silver_features_rt"
                )

            all_dfs.append(df)

        except Exception as e:
            log.error(f"  Error procesando {ticker}: {e}")

    return pd.concat(all_dfs) if all_dfs else pd.DataFrame()
