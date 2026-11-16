from __future__ import annotations

from pathlib import Path

from ....contracts import DataBlockLike
from ....templates import PipelineTemplate
from ....implements import DataBlock
from ....arrays import Array
from .core import StaticCoreTemplate


class StaticPipelineTemplate(PipelineTemplate):

    def fmu_export_artifacts(self, fmu_path: str | Path) -> None:
        fmu_dir: Path = Path(fmu_path)
        fmu_dir.mkdir(parents=True, exist_ok=True)
        fmu_resources_dir: Path = fmu_dir / "resources"
        fmu_resources_dir.mkdir(parents=True, exist_ok=True)
        self.coder_.save(path=fmu_resources_dir / "coder.bin")
        if isinstance(self.core_, StaticCoreTemplate):
            self.core_.fmu_export_artifacts(fmu_path=fmu_resources_dir)
            pass
        pass

    def rollout(self, datablock: DataBlockLike, *args, **kwargs) -> DataBlock:
        x: Array = datablock.x()
        y: Array = self.predict(x=x)
        return DataBlock(x, y)
        pass

    pass