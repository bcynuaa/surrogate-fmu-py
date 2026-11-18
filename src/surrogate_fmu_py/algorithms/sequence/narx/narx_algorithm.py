from __future__ import annotations

from typing import Dict, Any

from ....utils import format_dict
from ..shared import (
    SequenceWindowDatasetImplConfig, 
    SequenceWindowTrainerImplConfig,
    
    SequenceWindowTrainerImplConfig,

    SequenceAlgorithmTmpl,
)
from .narx_core import NARXCoreConfig, NARXCore



class NARXAlgorithm(SequenceAlgorithmTmpl):


    def create_core(self, params: Dict[str, Any]) -> NARXCore:
        core: NARXCore = NARXCore().init(params=params)
        return core
        pass

    def default_params(self) -> Dict[str, Any]:
        return DEFAULT_PARAMS_DICT.copy()
        pass

    def format_params(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = format_dict(self.default_params(), overrides)
        # * check activations
        hidden_len: int = len(params.get("hidden_layers", []))
        activations: Any = params.get("activations", None)
        if activations is not None:
            activations = list(activations)
            if len(activations) < hidden_len:
                activations = activations + ["relu"] * (hidden_len - len(activations))
                pass
            if len(activations) > hidden_len:
                activations = activations[:hidden_len]
                pass
            params["activations"] = activations
            pass
        else:
            params["activations"] = ["relu"] * hidden_len
            pass
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
        # * check window = max(wx_dim, wy_dim)
        wx_dim: int = params.get("wx_dim", 16)
        wy_dim: int = params.get("wy_dim", 8)
        params["window"] = max(wx_dim, wy_dim)
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
    **NARXCoreConfig().to_dict(),

    # * training settings
    **SequenceWindowTrainerImplConfig().to_dict(),
}
