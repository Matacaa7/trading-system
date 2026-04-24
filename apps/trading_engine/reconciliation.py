"""
reconciliation.py
─────────────────
Módulo NUEVO — resuelve F-39 del informe.

Consulta Alpaca periódicamente para detectar fills y cierres de bracket orders.
Actualiza gold_trades con ts_salida, precio_salida, pnl, motivo_salida reales.

Sin este módulo, gold_trades queda con operaciones eternamente "abiertas" y
el frontend no puede mostrar P&L realizado ni historial de cierres.

TODO Fase 3.5: implementar
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def reconcile_trades() -> int:
    """Revisa gold_trades con ts_salida IS NULL y actualiza con datos de Alpaca.

    Returns:
        Número de trades actualizados.
    """
    raise NotImplementedError("TODO Fase 3.5")
