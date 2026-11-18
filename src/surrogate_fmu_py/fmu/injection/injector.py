from __future__ import annotations

from typing import List, Dict
from pathlib import Path
import shutil

from .platform import Platform, get_platform
from .binary import FMUBinary, get_binary
from .runtime import RuntimeLibrary, get_runtime


class FMUInjector:

    def __init__(
        self,
        binary_name: str = "vector_coder_onnx",
        fmi_version: int = 2,
        fmi_interface: str = "cs",
        platforms: List[str] | None = None,
    ) -> None:
        self.binary_: FMUBinary = get_binary(binary_name)
        self.fmi_version_: int = int(fmi_version)
        if self.fmi_version_ not in self.binary_.supported_fmi_versions():
            raise ValueError(f"FMI version '{fmi_version}' is not supported by binary '{binary_name}'. Supported versions: {self.binary_.supported_fmi_versions()}")
            pass
        self.fmi_interface_: str = fmi_interface.lower().strip()
        if self.fmi_interface_ not in self.binary_.supported_fmi_interfaces():
            raise ValueError(f"FMI interface '{self.fmi_interface_}' is not supported by binary '{binary_name}'. Supported interfaces: {self.binary_.supported_fmi_interfaces()}")
            pass
        requested_raw = platforms if platforms is not None else ["win64-x64"]
        if not requested_raw:
            raise ValueError("At least one platform must be requested")
        requested: List[str] = list(dict.fromkeys(get_platform(name).name for name in requested_raw))
        self.platforms_: List[Platform] = [get_platform(name) for name in requested]
        binary_files = self.binary_.binary_files(fmi_version=self.fmi_version_, fmi_interface=self.fmi_interface_)
        missing = [name for name in requested if name not in binary_files]
        if missing:
            raise FileNotFoundError(f"Missing {binary_name} binary for FMI {self.fmi_version_} {self.fmi_interface_}: {missing}")
        folders = [platform.fmu_binary_folder(self.fmi_version_) for platform in self.platforms_]
        if len(set(folders)) != len(folders):
            raise ValueError("Requested platforms share an FMU binary folder for this FMI version")
        self.runtimes_: List[RuntimeLibrary] = [get_runtime(runtime_name) for runtime_name in self.binary_.required_runtimes()]
        for runtime in self.runtimes_:
            missing = [name for name in requested if name not in runtime.supported_platforms()]
            if missing:
                raise FileNotFoundError(f"Missing {runtime.runtime_name()} runtime for: {missing}")
            runtime_filenames: set[str] = set()
            for name in requested:
                filenames = {path.name for path in (runtime.path() / name).iterdir() if path.is_file()}
                if not filenames:
                    raise FileNotFoundError(f"Empty {runtime.runtime_name()} runtime directory for {name}")
                overlap = runtime_filenames & filenames
                if overlap:
                    raise ValueError(f"Runtime files collide across platforms: {sorted(overlap)}")
                runtime_filenames.update(filenames)
        pass

    def inject(self, fmu_dir: str | Path, model_identifier: str="model") -> None:
        binaries_dir: Path = Path(fmu_dir) / "binaries"
        binaries_dir.mkdir(parents=True, exist_ok=True)
        if not model_identifier:
            model_identifier = self.binary_.binary_name()
            pass
        binary_files: Dict[str, Path] = self.binary_.binary_files(fmi_version=self.fmi_version_, fmi_interface=self.fmi_interface_)
        for platform in self.platforms_:
            source_path: Path = binary_files[platform.name]
            target_path: Path = binaries_dir / platform.fmu_binary_folder(self.fmi_version_) / f"{model_identifier}{platform.fmu_binary_suffix()}"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            pass
        for runtime in self.runtimes_:
            runtime.inject(fmu_dir=fmu_dir, platforms=[platform.name for platform in self.platforms_], fmi_version=self.fmi_version_)
            pass
        pass

    pass
