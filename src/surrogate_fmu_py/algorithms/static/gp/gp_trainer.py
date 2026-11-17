from __future__ import annotations

from typing import Dict, Any, Self

from ....arrays import Array, to_numpy
from ....contracts import TrainerLike, CoderLike, CoreLike, DatasetLike, ReporterLike
from ....implements import NoneReporter
from ..shared import StaticFullDatasetImpl
from .gp_core import GPCore


class GPTrainer(TrainerLike):

    def __init__(self) -> None:
        pass

    def init(self, params: Dict[str, Any]) -> Self:
        return self
        pass

    def train(
        self,
        coder: CoderLike,
        core: CoreLike,
        train_dataset: DatasetLike,
        validation_dataset: DatasetLike,
        reporter: ReporterLike | None = None,
    ) -> GPCore:
        if reporter is None:
            reporter = NoneReporter()
            pass
        if not isinstance(core, GPCore):
            raise TypeError("core must be an instance of GPCore.")
            pass
        if not isinstance(train_dataset, StaticFullDatasetImpl):
            raise TypeError("train_dataset must be an instance of StaticFullDatasetImpl.")
            pass
        reporter.message("GP: Start Training, Prepare Components.")
        reporter.progress(0.0, message="GP: Preparing Training Components.")
        x_data: Array = to_numpy(coder.encode_input(train_dataset.x()))
        reporter.progress(25.0, message="GP: Encode Input Data.")
        y_data: Array = to_numpy(coder.encode_output(train_dataset.y()))
        reporter.progress(50.0, message="GP: Encode Output Data.")
        core.fit(x=x_data, y=y_data)
        reporter.progress(100.0, message="GP: Training Completed.")
        return core
        pass

    pass
