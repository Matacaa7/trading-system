"""
base.py
───────
Interfaz común para todos los modelos del sistema.
Tanto sklearn como pytorch implementan esta clase abstracta,
lo que permite cambiar de modelo en el yaml sin modificar
pipeline.py ni evaluate.py.

Métodos obligatorios:
    fit(X_train, y_train)          Entrena el modelo
    predict(X)                     Devuelve predicciones
    predict_proba(X)               Devuelve probabilidades (clasificación)
    confidence_interval(X, alpha)  Devuelve intervalo de confianza
    save(path)                     Guarda el modelo en disco
    load(path)                     Carga un modelo desde disco
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """
    Clase abstracta que todos los modelos deben implementar.
    Garantiza la misma API para sklearn y pytorch.
    """

    def __init__(self, task: str = "classification", **params):
        """
        Args:
            task:   'classification' o 'regression'
            params: hiperparámetros del modelo
        """
        self.task = task
        self.params = params
        self._model = None

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entrena el modelo con los datos de entrenamiento."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve predicciones.
        Clasificación: clases (0 o 1)
        Regresión: valores continuos
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve probabilidades de la clase positiva.
        Solo relevante para clasificación.
        Para regresión devuelve los mismos valores que predict().
        """
        ...

    @abstractmethod
    def confidence_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula el intervalo de confianza para las predicciones.

        Cada subclase decide la estrategia (bootstrap, percentiles,
        MC dropout, etc.). Documentar la estrategia usada en la docstring
        de la implementación concreta.

        Args:
            X:     features
            alpha: nivel de significancia (0.05 = intervalo 95%)

        Returns:
            (lower, upper) arrays con los límites del intervalo
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Guarda el modelo entrenado en disco."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, task: str = "classification") -> "BaseModel":
        """Carga un modelo entrenado desde disco."""
        ...

    @property
    def is_fitted(self) -> bool:
        """True si el modelo ya ha sido entrenado."""
        return self._model is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, params={self.params})"
