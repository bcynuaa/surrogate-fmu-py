from __future__ import annotations

from typing import Sequence, Dict, Any

from ....contracts import DataBlockLike
from ....utils import format_dict
from ..shared import StaticAlgorithmTmpl
from ..shared import StaticFullDatasetImpl
from .gp_core import GPCoreConfig, GPCore
from .gp_trainer import GPTrainer


class GPAlgorithm(StaticAlgorithmTmpl):

    def default_params(self) -> Dict[str, Any]:
        return DEFAULT_PARAMS_DICT.copy()
        pass

    def format_params(self, overrides: Dict[str, Any]={}) -> Dict[str, Any]:
        params: Dict[str, Any] = format_dict(self.default_params(), overrides)
        return params
        pass

    def create_core(self, params: Dict[str, Any]) -> GPCore:
        core: GPCore = GPCore().init(params=params)
        return core
        pass

    def create_dataset(self, datablocks: Sequence[DataBlockLike], params: Dict[str, Any]) -> StaticFullDatasetImpl:
        return StaticFullDatasetImpl.from_datablocks(datablocks=datablocks, params=params)
        pass

    def create_trainer(self, params: Dict[str, Any]) -> GPTrainer:
        trainer: GPTrainer = GPTrainer().init(params=params)
        return trainer
        pass

    pass


DEFAULT_PARAMS_DICT: Dict[str, Any] = {
    # * scaler settings
    "x_scaler_mode": "normal",
    "y_scaler_mode": "normal",

    # * model settings
    **GPCoreConfig().to_dict(),
}
