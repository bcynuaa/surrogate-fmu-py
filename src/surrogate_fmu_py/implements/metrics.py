from __future__ import annotations

from typing import List
from dataclasses import dataclass

import numpy as np

from ..contracts import MetricsLike
from ..arrays import Array


@dataclass(frozen=True)
class Metrics(MetricsLike):
    mae: float
    mse: float
    rmse: float
    r2: float
    nrmse: float
    pearsonr: float
    fidelity: float
    def fidelity_value(self) -> float:
        return self.fidelity
        pass
    pass


@dataclass(frozen=True)
class ChannelMetrics(MetricsLike):
    n: int
    overall: Metrics
    metrics: List[Metrics]
    def fidelity_value(self) -> float:
        return self.overall.fidelity
        pass
    pass


EPS: float = 1e-10


def compute_metrics(y_pred: Array, y_true: Array) -> ChannelMetrics:
    # y: (n_samples, n_channels)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if not y_pred.shape == y_true.shape:
        raise ValueError(f"y_pred and y_true must have the same shape, but got {y_pred.shape} and {y_true.shape}")
        pass
    error: Array = y_pred - y_true

    output_min: Array = np.min(y_true, axis=0)
    output_max: Array = np.max(y_true, axis=0)
    output_ranges: Array = output_max - output_min
    safe_ranges: Array = np.where(output_ranges <= EPS, 1.0, output_ranges)

    metrics_list: List[Metrics] = []
    for index in range(y_true.shape[-1]):
        true_channel: Array = y_true[:, index]
        pred_channel: Array = y_pred[:, index]
        error_channel: Array = error[:, index]
        
        mse: float = np.mean(error_channel ** 2)
        rmse: float = np.sqrt(mse)
        mae: float = np.mean(np.abs(error_channel))

        target_var: float = float(np.sum((true_channel - np.mean(true_channel)) ** 2))
        residual_sum: float = float(np.sum(error_channel ** 2))
        if target_var <= EPS:
            r2: float = 1.0 if residual_sum <= EPS else 0.0
            pass
        else:
            r2 = float(1.0 - (residual_sum / target_var))
            pass
        nrmse: float = rmse / float(safe_ranges[index])
        true_var: float = float(np.var(true_channel))
        pred_var: float = float(np.var(pred_channel))
        if true_var <= EPS or pred_var <= EPS:
            pearsonr: float = 1.0 if np.abs(true_var - pred_var) <= EPS else 0.0
            pass
        else:
            pearsonr = float(np.corrcoef(true_channel, pred_channel)[0, 1])
            pearsonr = np.clip(pearsonr, -1.0, 1.0)
            pass

        mean_error: float = float(np.mean(error_channel))
        bias_ratio: float = abs(mean_error) / float(safe_ranges[index])
        bias_penalty: float = float(np.clip(2.0 * bias_ratio, 0.0, 0.3))
        fidelity: float = 100.0 * max(0.0, 1.0 - nrmse) * max(0.0, pearsonr)
        fidelity *= 1.0 - bias_penalty
        fidelity = float(np.clip(fidelity, 0.0, 100.0))

        metrics_list.append(Metrics(mae=mae, mse=mse, rmse=rmse, r2=r2, nrmse=nrmse, pearsonr=pearsonr, fidelity=fidelity))
        pass
    overall_rmse: float = np.sqrt(np.mean(error ** 2))
    global_range: float = float(np.max(y_true) - np.min(y_true))
    if global_range <= EPS:
        global_range = float(np.max(safe_ranges))
        pass
    if global_range <= EPS:
        global_range = 1.0
        pass

    overall: Metrics = Metrics(
        mae=float(np.mean(np.abs(error))),
        mse=float(np.mean(error ** 2)),
        rmse=overall_rmse,
        r2=np.mean([m.r2 for m in metrics_list]),
        nrmse=overall_rmse / global_range,
        pearsonr=float(np.mean([m.pearsonr for m in metrics_list])),
        fidelity=float(np.mean([m.fidelity for m in metrics_list])),
    )
    return ChannelMetrics(n=y_true.shape[-1], overall=overall, metrics=metrics_list)
    pass