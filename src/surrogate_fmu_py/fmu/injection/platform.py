from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

# * This file only defines the platforms and their properties
# * which doesn't mean that all the platforms listed here are available from the artifacts


@dataclass(frozen=True)
class Platform:
    name: str
    fmi2_binary_folder: str
    fmi3_binary_folder: str
    fmi_binary_suffix: str
    runtime_suffixes: Tuple[str, ...]

    def fmu_binary_folder(self, major: int) -> str:
        major_int: int = int(major)
        if major_int == 2:
            return self.fmi2_binary_folder
            pass
        if major_int == 3:
            return self.fmi3_binary_folder
            pass
        raise ValueError(f"Unsupported FMI major version: {major}")
        pass

    def fmu_binary_suffix(self) -> str:
        return self.fmi_binary_suffix
        pass

    def os(self) -> str:
        return self.name.split("-")[0]
        pass

    def arch(self) -> str:
        return self.name.split("-")[1]
        pass

    def to_str(self) -> str:
        return self.name
        pass

    def platform_name(self) -> str:
        return self.to_str()
        pass

    def __str__(self) -> str:
        return self.to_str()
        pass

    def __repr__(self) -> str:
        return self.to_str()
        pass

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Platform):
            return self.name == other.name
            pass
        elif isinstance(other, str):
            return self.name == other
            pass
        return False
        pass

    pass


# * https://fmi-standard.org/docs/3.0.2/#fmu-distribution
PLATFORMS_DICT: Dict[str, Platform] = {
    "win64-x64": Platform(
        name="win64-x64",
        fmi2_binary_folder="win64",
        fmi3_binary_folder="x86_64-windows",
        fmi_binary_suffix=".dll",
        runtime_suffixes=(".dll",),
    ),
    "win64-arm64": Platform(
        name="win64-arm64",
        fmi2_binary_folder="win64",
        fmi3_binary_folder="aarch64-windows",
        fmi_binary_suffix=".dll",
        runtime_suffixes=(".dll",),
    ),
    "linux64-x64": Platform(
        name="linux64-x64",
        fmi2_binary_folder="linux64",
        fmi3_binary_folder="x86_64-linux",
        fmi_binary_suffix=".so",
        runtime_suffixes=(".so",),
    ),
    "linux64-aarch64": Platform(
        name="linux64-aarch64",
        fmi2_binary_folder="linux64",
        fmi3_binary_folder="aarch64-linux",
        fmi_binary_suffix=".so",
        runtime_suffixes=(".so",),
    ),
    "darwin64-arm64": Platform(
        name="darwin64-arm64",
        fmi2_binary_folder="darwin64",
        fmi3_binary_folder="aarch64-darwin",
        fmi_binary_suffix=".dylib",
        runtime_suffixes=(".dylib",),
    ),
}


def get_platform(name: str) -> Platform:
    name_str: str = name.lower().strip()
    if name_str not in PLATFORMS_DICT:
        raise ValueError(f"Platform '{name}' is not available. Available platforms: {list(PLATFORMS_DICT)}")
        pass
    return PLATFORMS_DICT[name_str]
    pass


def list_platforms() -> List[str]:
    return list(PLATFORMS_DICT.keys())
    pass


def filter_platforms(platforms: List[str]) -> List[str]:
    available_platforms: List[str] = list_platforms()
    filtered_platforms: List[str] = [p for p in platforms if p in available_platforms]
    # win64-x64, win64-arm64 -> win64-x64
    # linux64-x64, linux64-aarch64 -> linux64-x64
    if "win64-x64" in filtered_platforms and "win64-arm64" in filtered_platforms:
        filtered_platforms.remove("win64-arm64")
        pass
    if "linux64-x64" in filtered_platforms and "linux64-aarch64" in filtered_platforms:
        filtered_platforms.remove("linux64-aarch64")
        pass
    return filtered_platforms
    pass