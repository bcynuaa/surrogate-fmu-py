from __future__ import annotations

from typing import Self
from pathlib import Path

from ....contracts import CoreLike


class StaticCoreTemplate(CoreLike):

    def alg_family(self) -> str:
        return "static"
        pass

    def mode_train(self) -> Self:
        return self
        pass

    def mode_eval(self) -> Self:
        return self
        pass

    def fmu_export_artifacts(self, fmu_path: str | Path) -> None:
        return None
        pass

    pass