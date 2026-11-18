from __future__ import annotations

from typing import Dict, Any, Self
from pathlib import Path

from ....contracts import CoreLike
from ....arrays import Array


class StaticCoreTmpl(CoreLike):

    def alg_family(self) -> str: return "static"
    def mode_train(self) -> Self: return self
    def mode_eval(self) -> Self: return self
    def fmu_required_binary(self) -> str: return "vector_coder_onnx"

    def alg_name(self) -> str: raise NotImplementedError("Subclasses must implement the alg_name method.")
    def init(self, params: Dict[str, Any]) -> Self: raise NotImplementedError("Subclasses must implement the init method.")
    def predict(self, x: Array) -> Array: raise NotImplementedError("Subclasses must implement the predict method.")
    def save(self, path: str | Path) -> Self: raise NotImplementedError("Subclasses must implement the save method.")
    def load(self, path: str | Path) -> Self: raise NotImplementedError("Subclasses must implement the load method.")

    def fmu_export_core_artifacts(self, fmu_path: str | Path) -> None: raise NotImplementedError("Subclasses must implement the fmu_export_core_artifacts method.")

    pass