from __future__ import annotations

from typing import Sequence, Dict, Any

from ....contracts import DataBlockLike
from ..templates import StaticAlgorithmTemplate
from ..implements import StaticTorchDataset
from .core import DNNCoreConfig, DNNCore
from .trainer import DNNTrainerConfig, DNNTrainer


class DNNAlgorithm(StaticAlgorithmTemplate):

    def default_params(self) -> Dict[str, Any]:
        return DEFAULT_PARAMS_DICT.copy()
        pass

    def format_params(self, overrides: Dict[str, Any]={}) -> Dict[str, Any]:
        overrides = overrides.copy()
        params: Dict[str, Any] = self.default_params()
        params.update(overrides)
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
        return params
        pass

    def create_core(self, params: Dict[str, Any]) -> DNNCore:
        core: DNNCore = DNNCore().init(params=params)
        return core
        pass

    def create_dataset(self, datablocks: Sequence[DataBlockLike], params: Dict[str, Any]) -> StaticTorchDataset:
        return StaticTorchDataset.from_datablocks(datablocks=datablocks, params=params)
        pass

    def create_trainer(self, params: Dict[str, Any]) -> DNNTrainer:
        trainer: DNNTrainer = DNNTrainer().init(params=params)
        return trainer
        pass

    pass


DEFAULT_PARAMS_DICT: Dict[str, Any] = {
    # * scaler settings
    "x_scaler_mode": "normal",
    "y_scaler_mode": "normal",

    # * model settings
    **DNNCoreConfig().to_dict(),
    
    # * training settings
    **DNNTrainerConfig().to_dict(),
}