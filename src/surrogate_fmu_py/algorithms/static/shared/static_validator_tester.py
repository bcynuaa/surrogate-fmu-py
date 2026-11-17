from __future__ import annotations

from ....contracts import ValidatorLike, TesterLike, PipelineLike, DatasetLike
from ....implements import ChannelMetrics, compute_metrics
from ....arrays import Array
from .static_dataset import StaticDatasetTmpl


def compute_metrics_from_dataset(
    pipeline: PipelineLike,
    dataset: DatasetLike
) -> ChannelMetrics:
    if isinstance(dataset, StaticDatasetTmpl):
        x: Array = dataset.x()
        y: Array = dataset.y()
        pass
    else:
        raise ValueError(f"dataset must have method x() and y(), but got {type(dataset)}")
        pass
    y_pred: Array = pipeline.predict(x=x)
    metrics: ChannelMetrics = compute_metrics(y_true=y, y_pred=y_pred)
    return metrics
    pass


class StaticFullValidatorImpl(ValidatorLike):

    def validate(self, pipeline: PipelineLike, validation_dataset: DatasetLike) -> ChannelMetrics:
        return compute_metrics_from_dataset(pipeline=pipeline, dataset=validation_dataset)
        pass

    pass


class StaticFullTesterImpl(TesterLike):

    def test(self, pipeline: PipelineLike, test_dataset: DatasetLike) -> ChannelMetrics:
        return compute_metrics_from_dataset(pipeline=pipeline, dataset=test_dataset)
        pass

    pass