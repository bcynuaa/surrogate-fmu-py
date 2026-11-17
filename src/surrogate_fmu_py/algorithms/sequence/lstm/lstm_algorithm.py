from __future__ import annotations

from typing import Dict, Any

from ..shared import (
    SequenceWindowDatasetImplConfig, 
    SequenceWindowTrainerImplConfig,
    
    SequenceWindowTrainerImplConfig,

    SequenceAlgorithmTmpl,
)
from .lstm_core import LSTMCoreConfig, LSTMCore


class LSTMAlgorithm(SequenceAlgorithmTmpl):

    def create_core(self, params: Dict[str, Any]) -> LSTMCore:
        core: LSTMCore = LSTMCore().init(params=params)
        return core
        pass

    def default_params(self) -> Dict[str, Any]:
        return DEFAULT_PARAMS_DICT.copy()
        pass

    def format_params(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = self.default_params()
        params.update(overrides)
        # * check channel_weights
        output_dim: int = params.get("output_dim", 1)
        channel_weights: Any = params.get("channel_weights", None)
        if channel_weights is not None:
            channel_weights = list(channel_weights)
            if len(channel_weights) < output_dim:
                channel_weights = channel_weights + [1.0] * (output_dim - len(channel_weights))
                pass
            if len(channel_weights) > output_dim:
                channel_weights = channel_weights[:output_dim]
                pass
            params["channel_weights"] = channel_weights
            pass
        else:
            params["channel_weights"] = [1.0] * output_dim
            pass
        return params
        pass

    pass


DEFAULT_PARAMS_DICT: Dict[str, Any] = {
    # * scaler settings
    "x_scaler_mode": "normal",
    "y_scaler_mode": "normal",

    # * dataset settings
    **SequenceWindowDatasetImplConfig().to_dict(),

    # * model settings
    **LSTMCoreConfig().to_dict(),

    # * training settings
    **SequenceWindowTrainerImplConfig().to_dict(),
}