from __future__ import annotations

from typing import List

import numpy as np

from ....contracts import PipelineLike, DatasetLike, DataBlockLike
from ....implements import ChannelMetrics, compute_metrics
from ....arrays import Array, to_numpy
from .sequence_pipeline import SequenceWindowPipelineImpl
from .sequence_dataset import SequenceWindowDatasetImpl


class SequenceWindowValidatorImpl:

    def validate(self, pipeline: PipelineLike, validation_dataset: DatasetLike) -> ChannelMetrics:
        y_true: List[Array] = []
        y_pred: List[Array] = []
        if not isinstance(validation_dataset, SequenceWindowDatasetImpl):
            raise ValueError(f"validation_dataset must be SequenceWindowDatasetImpl, but got {type(validation_dataset)}")
            pass
        for i in range(len(validation_dataset)):
            input_tensor, output_tensor = validation_dataset[i]
            x: Array = to_numpy(input_tensor)
            y: Array = to_numpy(output_tensor)
            x = x[None, ...] if x.ndim == 2 else x
            y_pred_i: Array = pipeline.predict(x=x)
            y_true.append(y.flatten())
            y_pred.append(y_pred_i.flatten())
            pass
        y_true_arr: Array = np.stack(y_true, axis=0)
        y_pred_arr: Array = np.stack(y_pred, axis=0)
        return compute_metrics(y_true=y_true_arr, y_pred=y_pred_arr)
        pass

    pass


class SequenceWindowTesterImpl:

    def test(self, pipeline: PipelineLike, test_dataset: DatasetLike) -> ChannelMetrics:
        if not isinstance(pipeline, SequenceWindowPipelineImpl):
            raise ValueError(f"pipeline must be SequenceWindowPipelineImpl, but got {type(pipeline)}")
            pass
        y_true: List[Array] = []
        y_pred: List[Array] = []
        for i in range(test_dataset.block_len()):
            true_datablock: DataBlockLike = test_dataset.block(i)
            pred_datablock: DataBlockLike= pipeline.rollout(datablock=true_datablock, y0=true_datablock.y()[:pipeline.window_dim()])
            y_true.append(true_datablock.y()[pipeline.window_dim():])
            y_pred.append(pred_datablock.y()[pipeline.window_dim():])
            pass
        y_true_arr: Array = np.concatenate(y_true, axis=0)
        y_pred_arr: Array = np.concatenate(y_pred, axis=0)
        return compute_metrics(y_true=y_true_arr, y_pred=y_pred_arr)
        pass

    pass