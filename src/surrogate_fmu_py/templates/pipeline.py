from __future__ import annotations

from typing import Dict, Any, Self
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

from ..contracts import DataBlockLike, CoderLike, CoreLike, PipelineLike
from ..arrays import Array, to_numpy
from ..utils import time_stamp

MANIFEST_FILENAME: str = "manifest.json"


class PipelineTemplate(PipelineLike):
    CODER_FILENAME: str = "coder.bin"
    CORE_FILENAME: str = "core.cbin"

    def __init__(self) -> None:
        pass

    def init(self, coder: CoderLike, core: CoreLike, params: Dict[str, Any]) -> Self:
        self.coder_: CoderLike = coder.to_numpy()
        self.core_: CoreLike = core.mode_eval()
        self.params_: Dict[str, Any] = params.copy()
        self.time_stamp_: str = time_stamp()
        return self
        pass

    def predict(self, x: Array) -> Array:
        xb: Array = to_numpy(x=self.coder_.encode(x=x))
        yb: Array = self.core_.predict(x=xb)
        y: Array = to_numpy(x=self.coder_.decode(x=yb))
        return y
        pass

    def manifest(self) -> Dict[str, Any]:
        return {
            "alg_family": self.core_.alg_family(),
            "alg_name": self.core_.alg_name(),
            "generate_tools": "surrogate-fmu-py",
            "generate_time": self.time_stamp_,
            "input_dim": self.input_dim(),
            "output_dim": self.output_dim(),
            "artifacts": {
                "coder": self.CODER_FILENAME,
                "core": self.CORE_FILENAME,
            },
            "params": self.params_,
        }
        pass

    def save(self, path: str | Path)-> Self:
        # .surr is a zip file containing:
        # 1. manifest.json
        # 2. core.cbin (.npz, .h5, .pt, .pkl, etc.)
        # 3. coder.bin
        surr_path: Path = Path(path)
        surr_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory() as tmpdir:
            tmp_path: Path = Path(tmpdir)
            coder_path: Path = tmp_path / self.CODER_FILENAME
            core_path: Path = tmp_path / self.CORE_FILENAME
            self.coder_.save(path=coder_path)
            self.core_.save(path=core_path)
            manifest_path: Path = tmp_path / MANIFEST_FILENAME
            manifest: Dict[str, Any] = self.manifest()
            with open(manifest_path, "w") as file:
                json.dump(manifest, file, indent=4)
                pass
            with zipfile.ZipFile(surr_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(manifest_path, arcname=MANIFEST_FILENAME)
                zipf.write(coder_path, arcname=self.CODER_FILENAME)
                zipf.write(core_path, arcname=self.CORE_FILENAME)
                pass
            pass
        return self
        pass

    def input_dim(self) -> int:
        return self.coder_.input_dim()
        pass

    def output_dim(self) -> int:
        return self.coder_.output_dim()
        pass

    def fmu_export_artifacts(self, fmu_path: str | Path) -> None:
        raise NotImplementedError("fmu_export_artifacts is not implemented yet.")
        pass

    def rollout(self, datablock: DataBlockLike, y0: Array | None, rollout_settings: Dict[str, Any] | None) -> DataBlockLike:
        raise NotImplementedError("rollout is not implemented yet.")
        pass

    pass