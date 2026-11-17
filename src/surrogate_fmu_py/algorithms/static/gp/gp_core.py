from __future__ import annotations

from typing import Dict, Any, Self
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Kernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.multioutput import MultiOutputRegressor
import sklearn

from ....arrays import Array
from ..shared import StaticCoreTmpl


@dataclass(frozen=True)
class GPCoreConfig:
    kernel_type: str = "RBF"
    n_restarts_optimizer: int = 0
    alpha: float = 1e-6
    length_scale: float = 1.0
    nu: float = 1.5
    rq_alpha: float = 1.0
    noise_level: float = 1e-6
    random_state: int = 42

    @classmethod
    def from_dict(cls, config: Dict[str, Any]={}) -> GPCoreConfig:
        return cls(
            kernel_type=config.get("kernel_type", "RBF"),
            n_restarts_optimizer=config.get("n_restarts_optimizer", 0),
            alpha=config.get("alpha", 1e-6),
            length_scale=config.get("length_scale", 1.0),
            nu=config.get("nu", 1.5),
            rq_alpha=config.get("rq_alpha", 1.0),
            noise_level=config.get("noise_level", 1e-6),
            random_state=config.get("random_state", 42),
        )
        pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        pass

    pass


def make_kernel(config: GPCoreConfig) -> Kernel:
    kernel_type: str = config.kernel_type.strip()
    if kernel_type == "RBF":
        base: Kernel = RBF(length_scale=config.length_scale)
        pass
    elif kernel_type == "Matern":
        base = Matern(length_scale=config.length_scale, nu=config.nu)
        pass
    elif kernel_type == "RationalQuadratic":
        base = RationalQuadratic(
            length_scale=config.length_scale,
            alpha=config.rq_alpha,
        )
        pass
    else:
        raise ValueError(
            f"Unsupported gp kernel: {kernel_type!r}. "
            "Only 'RBF' / 'Matern' / 'RationalQuadratic' are supported."
        )
        pass
    return ConstantKernel(1.0) * base + WhiteKernel(
        noise_level=config.noise_level,
        noise_level_bounds=(1e-10, 1e1),
    )
    pass


class GPCore(StaticCoreTmpl):

    def __init__(self) -> None:
        self.config_: GPCoreConfig = GPCoreConfig()
        self.model_: MultiOutputRegressor | None = None
        pass

    def alg_name(self) -> str:
        return "gp"
        pass

    def init(self, params: Dict[str, Any]) -> Self:
        self.config_ = GPCoreConfig.from_dict(config=params)
        self.model_ = MultiOutputRegressor(
            GaussianProcessRegressor(
                kernel=make_kernel(config=self.config_),
                n_restarts_optimizer=self.config_.n_restarts_optimizer,
                alpha=self.config_.alpha,
                random_state=self.config_.random_state,
                normalize_y=False,
            )
        )
        return self
        pass

    def mode_train(self) -> Self:
        return self
        pass

    def mode_eval(self) -> Self:
        return self
        pass

    def fit(self, x: Array, y: Array) -> None:
        if self.model_ is None:
            raise RuntimeError("Model is not initialized. Call init() first.")
            pass
        if x.ndim != 2:
            raise ValueError(f"Expected x to be a 2D array, got {x.ndim}D.")
            pass
        if y.ndim != 2:
            raise ValueError(f"Expected y to be a 2D array, got {y.ndim}D.")
            pass
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: x has {x.shape[0]} rows, but y has {y.shape[0]} rows."
            )
            pass
        if x.shape[0] == 0:
            raise ValueError("Cannot fit an empty dataset.")
            pass
        self.model_.fit(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )
        pass

    def predict(self, x: Array) -> Array:
        if self.model_ is None or not hasattr(self.model_, "estimators_"):
            raise RuntimeError("Model is not fitted. Call init() then fit() first.")
            pass
        if x.ndim != 2:
            raise ValueError(f"Expected x to be a 2D array, got {x.ndim}D.")
            pass
        y: Array = self.model_.predict(np.asarray(x, dtype=np.float64))
        return np.asarray(y, dtype=np.float64)
        pass

    def save(self, path: str | Path) -> Self:
        if self.model_ is None or not hasattr(self.model_, "estimators_"):
            raise RuntimeError("GPCore.save requires a fitted model: call init() then fit()")
            pass
        model_path: Path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config_.to_dict(),
                "sklearn_version": sklearn.__version__,
                "model": self.model_,
            },
            model_path,
        )
        return self
        pass

    def load(self, path: str | Path) -> Self:
        model_path: Path = Path(path)
        checkpoint: Dict[str, Any] = joblib.load(model_path)
        saved_version: str = str(checkpoint["sklearn_version"])
        if saved_version != sklearn.__version__:
            raise RuntimeError(
                f"Checkpoint was written by scikit-learn {saved_version}, "
                f"current environment is {sklearn.__version__}. "
                "Predictions may drift silently across versions; re-train or align the environment."
            )
            pass
        config: Dict[str, Any] = checkpoint.get("config", {})
        self.init(params=config)
        self.model_ = checkpoint["model"]
        return self
        pass

    def fmu_export_artifacts(self, fmu_path: str | Path) -> None:
        if self.model_ is None or not hasattr(self.model_, "estimators_"):
            raise RuntimeError("Model is not fitted. Call init() then fit() first.")
            pass
        fmu_dir: Path = Path(fmu_path)
        fmu_dir.mkdir(parents=True, exist_ok=True)
        fmu_resources_dir: Path = fmu_dir / "resources"
        fmu_resources_dir.mkdir(parents=True, exist_ok=True)
        onnx_model_path: Path = fmu_resources_dir / "model.onnx"
        from .gp_onnx import build_onnx
        import onnx

        onnx_model: Any = build_onnx(
            core=self,
            input_name="input",
            output_name="y",
            input_dim=int(self.model_.n_features_in_),
        )
        onnx.save(onnx_model, onnx_model_path)
        pass

    pass
