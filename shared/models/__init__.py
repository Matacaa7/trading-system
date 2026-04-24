"""shared.models — modelos del ensemble y registry."""

from shared.models.base import BaseModel
from shared.models.registry import get_model, all_model_names

__all__ = ["BaseModel", "get_model", "all_model_names"]
