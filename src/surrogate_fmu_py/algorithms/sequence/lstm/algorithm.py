from __future__ import annotations

from typing import Dict, Any

from ..implements import WindowDatasetConfig, WindowTrainerConfig, WindowTrainer
from .core import LSTMCoreConfig, LSTMCore


DEFAULT_PARAMS_DICT: Dict[str, Any] = {
    # * scaler settings
    "x_scaler_mode": "normal",
    "y_scaler_mode": "normal",

    # * dataset settings
    **WindowDatasetConfig().to_dict(),

    # * model settings
    **LSTMCoreConfig().to_dict(),

    # * training settings
    **WindowTrainerConfig().to_dict(),
}