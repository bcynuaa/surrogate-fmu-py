from __future__ import annotations

from typing import Sequence, Dict, Any

from ....contracts import (
    AlgorithmLike, 
    DataBlockLike, 
    DatasetLike, 
    CoderLike, CoreLike,
    TrainerLike
)
from ....implements import Coder, Scaler, get_scaler, concat_scalers
from .sequence_dataset import SequenceWindowDatasetImpl
from .sequence_trainer import SequenceWindowTrainerImpl
from .sequence_pipeline import SequenceWindowPipelineImpl
from .sequence_validator_tester import SequenceWindowValidatorImpl, SequenceWindowTesterImpl


class SequenceAlgorithmTmpl(AlgorithmLike):

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
        input_scaler: Scaler = concat_scalers(x_scaler, y_scaler)
        output_scaler: Scaler = y_scaler
        return Coder().init(input_scaler=input_scaler, output_scaler=output_scaler)
        pass

    def create_pipeline(self, coder: CoderLike, core: CoreLike, params: Dict[str, Any]) -> SequenceWindowPipelineImpl:
        return SequenceWindowPipelineImpl().init(coder=coder, core=core, params=params)
        pass

    def create_validator(self, params: Dict[str, Any]) -> SequenceWindowValidatorImpl:
        validator: SequenceWindowValidatorImpl = SequenceWindowValidatorImpl()
        return validator
        pass

    def create_tester(self, params: Dict[str, Any]) -> SequenceWindowTesterImpl:
        tester: SequenceWindowTesterImpl = SequenceWindowTesterImpl()
        return tester
        pass

    def create_dataset(self, datablocks: Sequence[DataBlockLike], params: Dict[str, Any]) -> SequenceWindowDatasetImpl:
        dataset: SequenceWindowDatasetImpl = SequenceWindowDatasetImpl().init(params=params)
        for datablock in datablocks:
            dataset.feed(datablock=datablock)
            pass
        dataset.finalize()
        return dataset
        pass

    def create_trainer(self, params: Dict[str, Any]) -> SequenceWindowTrainerImpl:
        trainer: SequenceWindowTrainerImpl = SequenceWindowTrainerImpl().init(params=params)
        return trainer
        pass

    def default_params(self) -> Dict[str, Any]: ...
    def format_params(self, overrides: Dict[str, Any]) -> Dict[str, Any]: ...
    def create_core(self, params: Dict[str, Any]) -> CoreLike: ...
    
    pass