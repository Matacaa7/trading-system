"""
db.py
─────
Funciones de escritura en Supabase para el trading-engine.
Guarda señales, decisiones, logs y timings en las tablas gold_*.

Cambios respecto al original:
    - F-26: usa shared.db.sb (singleton) en vez de _get_sb() por función
    - F-32: save_log ahora recibe run_id para correlacionar con timings

Uso:
    from apps.trading_engine.db import save_signals, save_decision, save_log
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from shared.db import sb
from shared.utils.time import now_utc, utc_isoformat

log = logging.getLogger(__name__)


def save_signals(signals: list[dict]) -> None:
    """Guarda las señales de cada modelo en gold_signals."""
    if not signals:
        return
    try:
        sb.table("gold_signals").upsert(
            signals, on_conflict="ts,ticker,experiment_name"
        ).execute()
        log.info(f"  {len(signals)} señales guardadas en gold_signals")
    except Exception as e:
        log.error(f"Error guardando señales: {e}")


def save_decision(decision: dict, detalle: dict) -> None:
    """Guarda la decisión combinada en gold_decisions."""
    try:
        row = {
            "ts": decision["ts"],
            "ticker": decision["ticker"],
            "decision": decision["decision"],
            "score_final": decision["score_final"],
            "detalle_modelos": json.dumps(detalle),
            "ejecutada": decision.get("ejecutada", False),
            "motivo_rechazo": decision.get("motivo_rechazo", ""),
            "run_at": utc_isoformat(),
        }
        sb.table("gold_decisions").upsert(row, on_conflict="ts,ticker").execute()
        log.info(
            f"  Decisión guardada: {decision['ticker']} -> "
            f"{decision['decision']} ({decision['score_final']:.1f})"
        )
    except Exception as e:
        log.error(f"Error guardando decisión: {e}")


def save_log(
    duration_s: float,
    tickers_procesados: list[str],
    senales_generadas: int,
    ordenes_ejecutadas: int,
    errores: list[str],
    status: str,
    run_id: str = "",
) -> None:
    """Guarda el log de auditoría de cada ejecución en gold_logs."""
    try:
        sb.table("gold_logs").insert({
            "run_at": utc_isoformat(),
            "run_id": run_id,
            "duration_s": round(duration_s, 2),
            "tickers_procesados": json.dumps(tickers_procesados),
            "senales_generadas": senales_generadas,
            "ordenes_ejecutadas": ordenes_ejecutadas,
            "errores": json.dumps(errores),
            "status": status,
        }).execute()
        log.info(f"  Log guardado: {status} ({duration_s:.1f}s)")
    except Exception as e:
        log.error(f"Error guardando log: {e}")


def save_timing(
    fase: str,
    duration_s: float,
    run_id: str,
    ticker: str | None = None,
    status: str = "ok",
) -> None:
    """Guarda el timing de una fase del pipeline en gold_pipeline_timings."""
    try:
        sb.table("gold_pipeline_timings").insert({
            "run_at": utc_isoformat(),
            "run_id": run_id,
            "fase": fase,
            "duration_s": round(duration_s, 3),
            "ticker": ticker,
            "status": status,
        }).execute()
    except Exception as e:
        log.error(f"Error guardando timing de {fase}: {e}")
