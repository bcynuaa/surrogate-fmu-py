from __future__ import annotations

from typing import Sequence, Dict, Any

from ....contracts import (
    AlgorithmLike, 
    DataBlockLike, 
    DatasetLike, 
    CoderLike, CoreLike,
    TrainerLike
)
from ....implements import Coder, Scaler, get_scaler
from .static_validator_tester import StaticFullValidatorImpl, StaticFullTesterImpl
from .static_pipeline import StaticExternalCoderPipelineImpl


class StaticAlgorithmTmpl(AlgorithmLike):

    def create_coder(self, datablocks: Sequence[DataBlockLike], params: Dict[str, Any]) -> Coder:
        x_scaler_mode: str = params.get("x_scaler_mode", "normal")
        y_scaler_mode: str = params.get("y_scaler_mode", "normal")
        x_scaler: Scaler = get_scaler(scaler_mode=x_scaler_mode)
        y_scaler: Scaler = get_scaler(scaler_mode=y_scaler_mode)
        for datablock in datablocks:
            x_scaler.feed(x=datablock.x())
            y_scaler.feed(x=datablock.y() if datablock.y() is not None else datablock.x())
            pass
        x_scaler.finalize()
        y_scaler.finalize()
        return Coder().init(input_scaler=x_scaler, output_scaler=y_scaler)
        pass

    def create_pipeline(self, coder: CoderLike, core: CoreLike, params: Dict[str, Any]) -> StaticExternalCoderPipelineImpl:
        return StaticExternalCoderPipelineImpl().init(coder=coder, core=core, params=params)
        pass

    def create_validator(self, params: Dict[str, Any]) -> StaticFullValidatorImpl:
        validator: StaticFullValidatorImpl = StaticFullValidatorImpl()
        return validator
        pass

    def create_tester(self, params: Dict[str, Any]) -> StaticFullTesterImpl:
        tester: StaticFullTesterImpl = StaticFullTesterImpl()
        return tester
        pass

    def default_params(self) -> Dict[str, Any]: raise NotImplementedError(f"default_params() is not implemented in {type(self)}")
    def format_params(self, overrides: Dict[str, Any]) -> Dict[str, Any]: raise NotImplementedError(f"format_params() is not implemented in {type(self)}")
    def create_core(self, params: Dict[str, Any]) -> CoreLike: raise NotImplementedError(f"create_core() is not implemented in {type(self)}")
    def create_dataset(self, datablocks: Sequence[DataBlockLike], params: Dict[str, Any]) -> DatasetLike: raise NotImplementedError(f"create_dataset() is not implemented in {type(self)}")
    def create_trainer(self, params: Dict[str, Any]) -> TrainerLike: raise NotImplementedError(f"create_trainer() is not implemented in {type(self)}")

    pass