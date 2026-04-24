"""
registry.py
───────────
Conecta el nombre del modelo en el yaml con la clase Python correcta.
Consulta silver_model_library en Supabase para verificar que el modelo
está activo y obtener sus parámetros por defecto.

Cambios respecto al original (ml-sandbox/models/registry.py):
    - F-43: usa shared.db.sb en lugar de create_client() por llamada
    - F-47: log de warning cuando el yaml sobrescribe defaults
    - Imports adaptados al monorepo (shared.models.X)
    - API simplificada: parámetros explícitos en lugar de Config monolítico

Uso:
    from shared.models.registry import get_model

    model = get_model("xgboost", task="classification", params={"n_estimators": 200})
"""

from __future__ import annotations

import importlib
import logging

from shared.db import sb
from shared.models.base import BaseModel

log = logging.getLogger(__name__)

# ─── Mapeo nombre → clase Python ───────────────────────────────────
# sklearn: importación directa (son ligeros)
# pytorch: importación lazy via string para no cargar torch si no hace falta

SKLEARN_MODELS: dict[str, type[BaseModel]] = {}
PYTORCH_MODELS: dict[str, str] = {
    "lstm": "shared.models.pytorch_models.lstm_model.LSTMModel",
    "gru": "shared.models.pytorch_models.gru_model.GRUModel",
    "transformer": "shared.models.pytorch_models.transformer_model.TransformerModel",
}


def _init_sklearn_models() -> None:
    """Importa las clases sklearn la primera vez que se necesitan.

    Separado en función para que un import error en un modelo concreto
    no impida importar registry.py (útil si solo usas pytorch).
    """
    if SKLEARN_MODELS:
        return  # ya inicializado

    from shared.models.sklearn_models.xgboost_model import XGBoostModel
    from shared.models.sklearn_models.random_forest_model import RandomForestModel
    from shared.models.sklearn_models.lightgbm_model import LightGBMModel

    SKLEARN_MODELS.update({
        "xgboost": XGBoostModel,
        "random_forest": RandomForestModel,
        "lightgbm": LightGBMModel,
    })


def all_model_names() -> list[str]:
    """Lista de todos los nombres de modelo soportados."""
    _init_sklearn_models()
    return list(SKLEARN_MODELS.keys()) + list(PYTORCH_MODELS.keys())


def get_model(
    model_name: str,
    task: str = "classification",
    params: dict | None = None,
    check_library: bool = True,
    **extra_kwargs,
) -> BaseModel:
    """
    Instancia el modelo correcto según su nombre.

    1. (Opcional) Verifica que el modelo existe y está activo en silver_model_library
    2. Combina default_params de Supabase con los params recibidos (params tiene prioridad)
    3. Instancia la clase del modelo

    Args:
        model_name:     nombre del modelo (xgboost, lstm, etc.)
        task:           'classification' o 'regression'
        params:         hiperparámetros del yaml (tienen prioridad sobre defaults)
        check_library:  si True, consulta silver_model_library en Supabase
        **extra_kwargs: kwargs adicionales pasados al constructor (ej: cfg para pytorch)

    Returns:
        Instancia del modelo lista para entrenar
    """
    _init_sklearn_models()
    params = params or {}

    if model_name not in SKLEARN_MODELS and model_name not in PYTORCH_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' no reconocido. "
            f"Disponibles: {all_model_names()}"
        )

    # ─── Consultar silver_model_library (opcional) ──────────────
    default_params: dict = {}

    if check_library:
        try:
            resp = (
                sb.table("silver_model_library")
                .select("model_name, model_type, active, default_params")
                .eq("model_name", model_name)
                .single()
                .execute()
            )
            library_entry = resp.data
        except Exception as e:
            log.warning(
                f"No se pudo consultar silver_model_library para '{model_name}': {e}. "
                f"Continuando sin defaults de Supabase."
            )
            library_entry = None

        if library_entry:
            if not library_entry.get("active", False):
                raise ValueError(
                    f"Modelo '{model_name}' está desactivado en silver_model_library"
                )
            default_params = library_entry.get("default_params") or {}

    # ─── Combinar params: defaults + yaml (yaml gana) ──────────
    final_params = {**default_params, **params}

    # F-47: avisar cuando el yaml sobrescribe un default
    overridden = set(default_params.keys()) & set(params.keys())
    if overridden:
        for key in sorted(overridden):
            log.info(
                f"  Param '{key}': default={default_params[key]} → yaml={params[key]}"
            )

    log.info(f"Modelo: {model_name} | Task: {task} | Params: {final_params}")

    # ─── Instanciar ─────────────────────────────────────────────
    if model_name in SKLEARN_MODELS:
        model_class = SKLEARN_MODELS[model_name]
        return model_class(task=task, **final_params)

    # PyTorch — importación lazy
    module_path, class_name = PYTORCH_MODELS[model_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(task=task, **final_params, **extra_kwargs)
