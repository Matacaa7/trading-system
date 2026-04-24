"""
inference.py
────────────
Lógica de predicción unificada. Usada por backtest (apps/ml_sandbox/backtest.py)
y por el live (apps/trading_engine/predictor.py).

Resuelve F-64 del informe de Fase 2 (código triplicado).

TODO en Fase 3.4:
- Migrar predict_row de backtest.py y predict_ticker de predictor.py aquí
- Eliminar access a model._net privado (F-66)
- Leer seq_len del modelo en lugar de hardcoded=10 (F-65)
- Emitir warning cuando se rellenan features faltantes con 0 (F-70)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def predict_row(
    row: pd.Series,
    df_history: pd.DataFrame,
    modelos: list[dict],
) -> dict:
    """Genera predicciones ensamblando los modelos cargados.
    
    Returns:
        dict con score_final, detalle por modelo, signals para BBDD
    """
    raise NotImplementedError("TODO Fase 3.4: migrar desde predictor.py + backtest.py")


def load_models(cfg: dict) -> list[dict]:
    """Carga modelos activos desde silver_model_registry."""
    raise NotImplementedError("TODO Fase 3.4: migrar desde predictor.py + backtest.py")
