from __future__ import annotations

from ....contracts import TesterLike, PipelineLike, DatasetLike
from ....implements import ChannelMetrics, compute_metrics
from ....arrays import Array
from .dataset import StaticFullDataset, StaticTorchDataset

class StaticFullTester(TesterLike):

    def test(
        self,
        pipeline: PipelineLike, 
        test_dataset: DatasetLike
    ) -> ChannelMetrics:
        if isinstance(test_dataset, StaticFullDataset) or isinstance(test_dataset, StaticTorchDataset):
            x: Array = test_dataset.x()
            y: Array = test_dataset.y()
            pass
        else:
            raise ValueError(f"test_dataset must have method x() and y(), but got {type(test_dataset)}")
            pass
        y_pred: Array = pipeline.predict(x=x)
        metrics: ChannelMetrics = compute_metrics(y_true=y, y_pred=y_pred)
        return metrics
        pass

    pass