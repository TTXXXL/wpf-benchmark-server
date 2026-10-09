"""Forecast model interfaces and built-in baselines."""

from .base import BaseForecaster
from .registry import MODEL_REGISTRY, get_model, register_model
from .persistence import PersistenceForecaster
from .naive import (SeasonalPersistenceForecaster, ClimatologyForecaster,
                    TrendPersistenceForecaster, FarmMeanPersistenceForecaster)
from .linear import LinearForecaster
from .increment import LinearIncrementForecaster, TCNIncrementForecaster
from .gbdt import GBDTForecaster
from .lstm_seq2seq import LSTMSeq2SeqForecaster
from .patchtst import PatchTSTForecaster
from .agcrn import AGCRNForecaster
from .agcrn_lite import AGCRNLiteForecaster
from .pin import PINForecaster, VARIANTS

__all__ = ["BaseForecaster", "MODEL_REGISTRY", "get_model", "register_model",
           "PersistenceForecaster", "SeasonalPersistenceForecaster",
           "ClimatologyForecaster", "TrendPersistenceForecaster",
           "FarmMeanPersistenceForecaster", "LinearForecaster", "GBDTForecaster",
           "LinearIncrementForecaster", "TCNIncrementForecaster",
           "LSTMSeq2SeqForecaster", "PatchTSTForecaster", "AGCRNForecaster",
           "AGCRNLiteForecaster",
           "PINForecaster", "VARIANTS"]
