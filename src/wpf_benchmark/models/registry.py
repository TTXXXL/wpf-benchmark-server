"""Explicit registry of forecast model classes."""
from __future__ import annotations

from typing import Dict, Type

from .base import BaseForecaster


MODEL_REGISTRY: Dict[str, Type[BaseForecaster]] = {}


def register_model(name: str):
    def decorator(cls: Type[BaseForecaster]) -> Type[BaseForecaster]:
        if not issubclass(cls, BaseForecaster):
            raise TypeError("Registered models must inherit BaseForecaster")
        if name in MODEL_REGISTRY:
            raise ValueError("Model already registered: " + name)
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> Type[BaseForecaster]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError("Unknown model '{}'; available: {}".format(
            name, ", ".join(sorted(MODEL_REGISTRY)))) from exc
