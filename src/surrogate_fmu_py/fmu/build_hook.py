from __future__ import annotations

from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        # * The wheel contains Windows and Linux native libraries. A universal
        # * wheel would also install on platforms that cannot load either one.
        build_data["tag"] = "py3-none-win_amd64.linux_x86_64"
        build_data["pure_python"] = False
        pass

    pass
