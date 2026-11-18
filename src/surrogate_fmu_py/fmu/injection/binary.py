from __future__ import annotations

from typing import List, Tuple, Dict, Protocol
from pathlib import Path

from .platform import Platform, list_platforms, get_platform


CURRENT_DIR: Path = Path(__file__).parent
BINARY_DIR: Path = CURRENT_DIR / "artifacts" / "binary"
if not BINARY_DIR.exists():
    raise FileNotFoundError(f"Binary artifacts directory {BINARY_DIR} does not exist.")
    pass


def get_available_platforms() -> List[str]:
    # support hot reload of binary artifacts
    return [item.name for item in list(BINARY_DIR.iterdir()) if item.is_dir() and item.name in list_platforms()]
    pass


class FMUBinary(Protocol):

    def binary_name(self) -> str: ...
    def require_initial_values(self) -> bool: ...
    def required_resources(self) -> Tuple[str, ...]: ...
    def required_runtimes(self) -> Tuple[str]: ...
    def supported_fmi_interfaces(self) -> Tuple[str]: ...
    def can_handle_variable_communication_step_size(self) -> bool: ...

    def supported_fmi_versions(self) -> List[int]:
        # walk through the binary artifacts directory and find the supported FMI versions
        supported_fmi_versions: List[int] = []
        for platform_name in get_available_platforms():
            platform_dir: Path = BINARY_DIR / platform_name
            for item in platform_dir.iterdir():
                if item.is_file() and item.name.startswith(f"fmi"):
                    parts: List[str] = item.name.split("_")
                    if len(parts) >= 3:
                        version_str: str = parts[0][3:]  # remove "fmi" prefix
                        try:
                            version: int = int(version_str)
                            if version not in supported_fmi_versions:
                                supported_fmi_versions.append(version)
                                pass
                            pass
                        except ValueError:
                            pass
                        pass
                    pass
                pass
            pass
        return supported_fmi_versions
        pass

    def supported_platforms(self) -> List[str]:
        # walk through the binary artifacts directory and find the supported platforms
        supported_platforms: List[str] = []
        for platform_name in get_available_platforms():
            platform_dir: Path = BINARY_DIR / platform_name
            for item in platform_dir.iterdir():
                if item.is_file() and item.name.startswith(f"fmi"):
                    if platform_name not in supported_platforms:
                        supported_platforms.append(platform_name)
                        pass
                    pass
                pass
            pass
        return supported_platforms
        pass

    def binary_files(self, fmi_version: int=2, fmi_interface: str="cs") -> Dict[str, Path]:
        keys: List[str] = self.supported_platforms()
        binary_files: Dict[str, Path] = {}
        fmi_version= int(fmi_version)
        fmi_interface = fmi_interface.lower().strip()
        for key in keys:
            binary_parent_path: Path = BINARY_DIR / key
            binary_file_basename: str = f"fmi{fmi_version}_{fmi_interface}_{self.binary_name()}"
            platform: Platform = get_platform(key)
            suffix: str = platform.fmi_binary_suffix
            binary_file_path: Path = binary_parent_path / f"{binary_file_basename}{suffix}"
            if binary_file_path.exists():
                binary_files[key] = binary_file_path
                pass
            pass
        if not binary_files:
            raise FileNotFoundError(f"No binary files found for {self.binary_name()} with FMI version {fmi_version} and interface {fmi_interface}.")
            pass
        return binary_files # {platform(key): path(value)}
        pass

    pass


class VectorCoderONNX(FMUBinary):
    def binary_name(self) -> str: return "vector_coder_onnx"
    def require_initial_values(self) -> bool: return False
    def required_resources(self) -> Tuple[str, ...]: return ("model.onnx", "coder.bin")
    def required_runtimes(self) -> Tuple[str]: return ("onnxruntime",)
    def supported_fmi_interfaces(self) -> Tuple[str]: return ("cs",)
    def can_handle_variable_communication_step_size(self) -> bool: return True
    pass


class WindowPureONNX(FMUBinary):
    def binary_name(self) -> str: return "window_pure_onnx"
    def require_initial_values(self) -> bool: return True
    def required_resources(self) -> Tuple[str, ...]: return ("model.onnx",)
    def required_runtimes(self) -> Tuple[str]: return ("onnxruntime",)
    def supported_fmi_interfaces(self) -> Tuple[str]: return ("cs",)
    def can_handle_variable_communication_step_size(self) -> bool: return False
    pass


FMU_BINARY_DICT: Dict[str, FMUBinary] = {
    item.binary_name(): item for item in [
        VectorCoderONNX(),
        WindowPureONNX(),
    ]
}


def get_binary(name: str) -> FMUBinary:
    name_str: str = name.lower().strip()
    if name_str not in FMU_BINARY_DICT:
        raise ValueError(f"Binary '{name}' is not available. Available binaries: {list(FMU_BINARY_DICT)}")
        pass
    return FMU_BINARY_DICT[name_str]
    pass


def get_available_binaries() -> List[str]:
    # support hot reload of binary artifacts
    return list(FMU_BINARY_DICT.keys())
    pass
