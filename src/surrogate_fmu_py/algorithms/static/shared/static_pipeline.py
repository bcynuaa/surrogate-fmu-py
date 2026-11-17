from __future__ import annotations

from pathlib import Path

from ....contracts import DataBlockLike
from ....templates import PipelineTmpl
from ....implements import DataBlock
from ....arrays import Array
from .static_core import StaticCoreTmpl


class StaticExternalCoderPipelineImpl(PipelineTmpl):

    def fmu_export_artifacts(self, fmu_path: str | Path) -> None:
        fmu_dir: Path = Path(fmu_path)
        fmu_dir.mkdir(parents=True, exist_ok=True)
        fmu_resources_dir: Path = fmu_dir / "resources"
        fmu_resources_dir.mkdir(parents=True, exist_ok=True)
        self.coder_.save(path=fmu_resources_dir / "coder.bin")
        if isinstance(self.core_, StaticCoreTmpl):
            self.core_.fmu_export_artifacts(fmu_path=fmu_path)
            pass
        pass

    def rollout(self, datablock: DataBlockLike, *args, **kwargs) -> DataBlock:
        x: Array = datablock.x()
        y: Array = self.predict(x=x)
        return DataBlock(x, y)
        pass

    pass