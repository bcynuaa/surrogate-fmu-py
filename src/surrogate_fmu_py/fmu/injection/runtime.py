from __future__ import annotations

from typing import List, Dict, Protocol
from pathlib import Path
import shutil

CURRENT_DIR: Path = Path(__file__).parent
RUNTIME_DIR: Path = CURRENT_DIR / "artifacts" / "runtime"

from .platform import list_platforms


class RuntimeLibrary(Protocol):
    def runtime_name(self) -> str: ...
    def inject(self, fmu_dir: str | Path, platforms: List[str], fmi_version: int = 2) -> None: ...

    def path(self) -> Path:
        return RUNTIME_DIR / self.runtime_name()
        pass

    def supported_platforms(self) -> List[str]:
        runtime_path: Path = self.path()
        if not runtime_path.exists():
            raise FileNotFoundError(f"Runtime library {runtime_path} does not exist.")
            pass
        folders_in_path: List[str] = [item.name for item in runtime_path.iterdir() if item.is_dir()]
        folders_available: List[str] = [folder for folder in folders_in_path if folder in list_platforms()]
        return sorted(folders_available)
        pass
    pass


class ONNXRuntimeLibrary(RuntimeLibrary):
    
    def runtime_name(self) -> str: return "onnxruntime"

    def inject(self, fmu_dir: str | Path, platforms: List[str], fmi_version: int = 2) -> None:
        platforms_to_inject: List[str] = list(dict.fromkeys(platforms))
        missing = [platform for platform in platforms_to_inject if platform not in self.supported_platforms()]
        if not platforms_to_inject or missing:
            raise ValueError(f"Unsupported ONNX Runtime platforms: {missing or platforms_to_inject}")
            pass
        fmu_dir_path: Path = Path(fmu_dir)
        if not fmu_dir_path.exists():
            fmu_dir_path.mkdir(parents=True, exist_ok=True)
            pass
        fmu_resources_dir: Path = fmu_dir_path / "resources"
        fmu_resources_dir.mkdir(parents=True, exist_ok=True)
        target_dir: Path = fmu_resources_dir
        for platform in platforms_to_inject:
            source_dir: Path = self.path() / platform
            # * warning: this will overwrite existing files in the target directory if they have the same name
            # * maybe we will put the runtime libraries in a subfolder named after the platform to avoid conflicts
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
            pass
        pass

    pass


RUNTIME_LIBRARY_DICT: Dict[str, RuntimeLibrary] = {
    item.runtime_name(): item for item in [
        ONNXRuntimeLibrary(),
    ]
}


def get_runtime(name: str) -> RuntimeLibrary:
    name_str: str = name.lower().strip()
    if name_str not in RUNTIME_LIBRARY_DICT:
        raise ValueError(f"Runtime library '{name}' is not available. Available runtimes: {list(RUNTIME_LIBRARY_DICT)}")
        pass
    return RUNTIME_LIBRARY_DICT[name_str]
    pass


def get_available_runtimes() -> List[str]:
    return list(RUNTIME_LIBRARY_DICT.keys())
    pass
