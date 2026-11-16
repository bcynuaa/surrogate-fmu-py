from __future__ import annotations

from typing import Dict, Any
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

from .contracts import PipelineLike, AlgorithmLike, CoderLike, CoreLike
from .registry import get_algorithm
from .implements import Coder

def load(path: str | Path) -> PipelineLike:
    surr_path: Path = Path(path)
    with TemporaryDirectory() as tmpdir:
        tmp_path: Path = Path(tmpdir)
        with zipfile.ZipFile(surr_path, "r") as zipf:
            zipf.extractall(tmp_path)
            pass
        manifest_path: Path = tmp_path / "manifest.json"
        with open(manifest_path, "r") as file:
            manifest: Dict[str, Any] = json.load(file)
            pass
        alg_name: str = manifest.get("alg_name", "unknown")
        artifacts_dict: Dict[str, Any] = manifest.get("artifacts", {})
        coder_path: Path = tmp_path / artifacts_dict.get("coder", "coder.bin")
        core_path: Path = tmp_path / artifacts_dict.get("core", "core.cbin")
        params: Dict[str, Any] = manifest.get("params", {}).copy()
        # * recover
        alg: AlgorithmLike = get_algorithm(alg_name=alg_name)
        coder: CoderLike = Coder().load(path=coder_path)
        core: CoreLike = alg.create_core(params=params).load(path=core_path)
        pass
    # * return the recovered pipeline
    return alg.create_pipeline(coder=coder, core=core, params=params)
    pass