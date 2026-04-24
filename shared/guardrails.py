"""
guardrails.py
─────────────
check_guardrails() unificado. Usado tanto por el backtest (apps/ml_sandbox/backtest.py)
como por el live (apps/trading_engine/engine.py).

Resuelve F-64 del informe de Fase 2 (código triplicado).

TODO en Fase 3.4: migrar aquí la implementación actual de engine.py
(es la versión más completa: tiene atr_volatilidad y circuit_breaker,
que backtest.py NO tiene — ver F-17).
"""

from __future__ import annotations

import pandas as pd


def check_guardrails(
    row: pd.Series,
    score: float,
    cfg_gr: dict,
    estado: dict,
) -> tuple[bool, str]:
    """Evalua los guardarraíles para una vela.
    
    Args:
        row: última vela de silver_features_rt (o silver_features para backtest)
        score: score ponderado de los modelos (0-100)
        cfg_gr: configuración de guardarraíles del yaml
        estado: estado actual del portfolio
    
    Returns:
        (pasa, motivo_rechazo)
    """
    raise NotImplementedError("TODO Fase 3.4: migrar desde engine.py")
