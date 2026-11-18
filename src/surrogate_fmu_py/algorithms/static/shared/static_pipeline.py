from __future__ import annotations

from typing import Dict, Any, Sequence
from pathlib import Path

from ....contracts import DataBlockLike
from ....templates import PipelineTmpl
from ....implements import DataBlock
from ....arrays import Array
from .static_core import StaticCoreTmpl


class StaticExternalCoderPipelineImpl(PipelineTmpl):

    def fmu_export_artifacts(self, fmu_path: str | Path) -> None:
        super().fmu_export_artifacts(fmu_path=fmu_path)
        fmu_dir: Path = Path(fmu_path)
        fmu_dir.mkdir(parents=True, exist_ok=True)
        fmu_resources_dir: Path = fmu_dir / "resources"
        fmu_resources_dir.mkdir(parents=True, exist_ok=True)
        self.coder_.save(path=fmu_resources_dir / "coder.bin")
        if isinstance(self.core_, StaticCoreTmpl):
            self.core_.fmu_export_core_artifacts(fmu_path=fmu_path)
            pass
        pass

    def rollout(self, datablock: DataBlockLike, *args, **kwargs) -> DataBlock:
        x: Array = datablock.x()
        y: Array = self.predict(x=x)
        return DataBlock(x, y)
        pass

    def fmu_export_spec(self, input_names: Sequence[str] | None=None, output_names: Sequence[str] | None=None) -> Dict[str, Any]:
        return super().fmu_export_spec(input_names=input_names, output_names=output_names)
        pass

    pass