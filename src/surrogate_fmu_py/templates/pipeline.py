from __future__ import annotations

from typing import Sequence, Dict, Any, Self
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

from ..contracts import DataBlockLike, CoderLike, CoreLike, PipelineLike
from ..arrays import Array, to_numpy
from ..utils import time_stamp, get_token, format_name_list

MANIFEST_FILENAME: str = "manifest.json"


class PipelineTmpl(PipelineLike):
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

    def fmu_export_spec(self, input_names: Sequence[str] | None=None, output_names: Sequence[str] | None=None) -> Dict[str, Any]:
        input_names = format_name_list(self.input_dim(), input_names, lambda index: f"x_{index+1}")
        output_names = format_name_list(self.output_dim(), output_names, lambda index: f"y_{index+1}")
        return {
            "binary": self.core_.fmu_required_binary(),
            "token": get_token(json.dumps(self.manifest(), indent=4)),
            "input_names": input_names,
            "output_names": output_names,
            "require_initial_values": False,
            "initial_names": None,
            "initial_values": None,
            "export_artifacts_func": self.fmu_export_artifacts,
        }
        pass

    def fmu_export_artifacts(self, fmu_path: str | Path) -> None:
        fmu_path_dir: Path = Path(fmu_path)
        fmu_path_dir.mkdir(parents=True, exist_ok=True)
        fmu_documentation_path: Path = fmu_path_dir / "documentation"
        fmu_documentation_path.mkdir(parents=True, exist_ok=True)
        with open(fmu_documentation_path / "manifest.json", "w") as file:
            json.dump(self.manifest(), file, indent=4)
            pass
        pass

    def rollout(self, datablock: DataBlockLike, y0: Array | None, rollout_settings: Dict[str, Any] | None) -> DataBlockLike:
        raise NotImplementedError("rollout is not implemented yet.")
        pass

    pass