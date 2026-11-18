from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import isfinite
from typing import Dict, List

from . import fmi2, fmi3
from .variable import FMIVariable, build_variables


# * Allowed FMI interface values. Each exported model selects exactly one.
VALID_FMI_INTERFACES: tuple[str, ...] = ("cs", "me")


@dataclass(frozen=True)
class FMIModelDescription:
    '''
    Everything modelDescription.xml needs, independent of the FMI version.

    token is the instantiation credential: FMI 2.0 calls it "guid", FMI 3.0
    calls it "instantiationToken". The exporter supplies its value.

    fmi_interface selects one interface. "cs" is implemented; "me" raises until
    the C++ side grows a model exchange entry point.

    State topology fields are reserved for ModelExchange. They are rejected
    until that interface is implemented.
    '''

    model_name: str
    model_identifier: str
    token: str
    variables: List[FMIVariable]
    fmi_version: int = 2
    fmi_interface: str = "cs"
    description: str | None = None
    generation_tool: str = "surrogate-fmu-py"
    variable_naming_convention: str = "flat"
    step_size: float | None = None
    can_handle_variable_communication_step_size: bool = True
    derivatives: Dict[int, int] = field(default_factory=dict)
    continuous_state_derivatives: List[int] = field(default_factory=list)
    event_indicators: List[int] = field(default_factory=list)

    def check(self) -> None:
        if self.fmi_version not in (2, 3):
            raise NotImplementedError(f"FMI {self.fmi_version} is not supported yet")
            pass
        if not self.model_name:
            raise ValueError("model_name must not be empty")
            pass
        if not self.model_identifier:
            raise ValueError("model_identifier must not be empty")
            pass
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.model_identifier) is None:
            raise ValueError(f"invalid model_identifier {self.model_identifier!r}")
            pass
        if not self.token:
            raise ValueError("token must not be empty")
            pass
        if self.variable_naming_convention not in ("flat", "structured"):
            raise ValueError(f"invalid variableNamingConvention {self.variable_naming_convention!r}")
            pass
        if self.fmi_interface not in VALID_FMI_INTERFACES:
            raise ValueError(f"unknown FMI interface {self.fmi_interface!r}, allowed: {VALID_FMI_INTERFACES}")
            pass
        if self.fmi_interface == "me":
            raise NotImplementedError("ModelExchange is not implemented yet")
            pass
        if self.derivatives or self.continuous_state_derivatives or self.event_indicators:
            raise NotImplementedError("state topology and event indicators are not implemented for CoSimulation")
            pass
        if self.step_size is not None and (not isfinite(self.step_size) or self.step_size <= 0):
            raise ValueError(f"step_size must be positive and finite, got {self.step_size!r}")
            pass
        if type(self.can_handle_variable_communication_step_size) is not bool:
            raise TypeError("can_handle_variable_communication_step_size must be bool")
            pass
        if not self.variables:
            raise ValueError("an FMU needs at least one variable")
            pass
        if not self.outputs():
            raise ValueError("an FMU needs at least one output")
            pass

        references: List[int] = [variable.value_reference for variable in self.variables]
        if len(set(references)) != len(references):
            raise ValueError(f"duplicated valueReference: {sorted(references)}")
            pass
        names: List[str] = [variable.name for variable in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicated variable name: {sorted(names)}")
            pass
        for variable in self.variables:
            variable.check()
            pass
        if self.fmi_version == 3:
            independent = [variable for variable in self.variables if variable.causality == "independent"]
            if len(independent) != 1:
                raise ValueError("FMI 3.0 requires exactly one independent variable")
                pass
            if independent[0].type != "Float64":
                raise ValueError("the independent time variable must have type Float64")
                pass
            pass
        pass

    def inputs(self) -> List[FMIVariable]:
        return [variable for variable in self.variables if variable.causality == "input"]
        pass

    def outputs(self) -> List[FMIVariable]:
        return [variable for variable in self.variables if variable.causality == "output"]
        pass

    def parameters(self) -> List[FMIVariable]:
        return [variable for variable in self.variables if variable.causality == "parameter"]
        pass

    def initial_dependencies(self) -> List[FMIVariable]:
        '''
        Variables that drive the outputs during initialization.

        Window and NeuralODE models initialize outputs from parameters, while
        the vector coder model runs one step using inputs.
        The FMI writers number this set differently (FMI 2.0 uses the 1-based
        position, FMI 3.0 the valueReference), so only the set lives here.
        '''
        return self.parameters() or self.inputs()
        pass

    pass


def build_model_description(model: FMIModelDescription) -> bytes:
    model.check()
    if model.fmi_version == 2:
        return fmi2.render(model)
        pass
    if model.fmi_version == 3:
        return fmi3.render(model)
        pass
    raise NotImplementedError(f"FMI {model.fmi_version} is not supported yet")
    pass


__all__ = [
    "FMIModelDescription",
    "FMIVariable",
    "VALID_FMI_INTERFACES",
    "build_model_description",
    "build_variables",
]
