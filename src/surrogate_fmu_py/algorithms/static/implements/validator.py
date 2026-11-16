from __future__ import annotations

from ....contracts import ValidatorLike, PipelineLike, DatasetLike
from ....implements import ChannelMetrics, compute_metrics
from ....arrays import Array
from .dataset import StaticFullDataset, StaticTorchDataset

class StaticFullValidator(ValidatorLike):

    def validate(
        self,
        pipeline: PipelineLike, 
        validation_dataset: DatasetLike
    ) -> ChannelMetrics:
        if isinstance(validation_dataset, StaticFullDataset) or isinstance(validation_dataset, StaticTorchDataset):
            x: Array = validation_dataset.x()
            y: Array = validation_dataset.y()
            pass
        else:
            raise ValueError(f"validation_dataset must have method x() and y(), but got {type(validation_dataset)}")
            pass
        y_pred: Array = pipeline.predict(x=x)
        metrics: ChannelMetrics = compute_metrics(y_true=y, y_pred=y_pred)
        return metrics
        pass

    pass