from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import List, Sequence

from .utils import format_start


# * Union of what the supported FMI versions accept. FMI 2.0 has six of them
# * (no structuralParameter), FMI 3.0 has seven. The per-version writer decides
# * what is actually emittable.
CAUSALITIES: List[str] = [
    "parameter",
    "calculatedParameter",
    "input",
    "output",
    "local",
    "independent",
    "structuralParameter",
]
VARIABILITIES: List[str] = ["constant", "fixed", "tunable", "discrete", "continuous"]
INITIALS: List[str] = ["exact", "approx", "calculated"]

# * Canonical type names follow FMI 3.0; the FMI 2.0 writer maps them back
# * (Float64 -> Real, Int32 -> Integer). Only scalar types are listed.
TYPES: List[str] = ["Float64", "Float32", "Int32", "Int64", "Boolean", "String"]


@dataclass(frozen=True)
class FMIVariable:
    '''
    One scalar variable of an FMU, expressed without any FMI version specifics.

    Shared by FMI 2.0 and FMI 3.0: name, value_reference, causality,
    variability, type, initial, start, description. Version specific pieces stay
    out of this class on purpose -- notably the FMI 2.0 ModelExchange
    "derivative" attribute, which FMI 3.0 replaces by an ordered
    <ContinuousStateDerivative> list. State topology is a ModelStructure
    concern, so it lives on FMIModelDescription instead.

    value_reference must stay in sync with fmu-library-cpp: it starts from 1 and
    the C++ side indexes values_[value_reference - 1]. The FMI 2.0 model
    structure is a different thing: it refers to the 1-based *position* of the
    variable in ModelVariables, which equals value_reference only while the
    references stay dense and ordered.
    '''

    name: str
    value_reference: int
    causality: str
    variability: str = "continuous"
    type: str = "Float64"
    initial: str | None = None
    start: float | int | bool | str | None = None
    description: str | None = None

    def check(self) -> None:
        if not self.name:
            raise ValueError("variable name must not be empty")
            pass
        if self.value_reference < 1:
            raise ValueError(f"valueReference starts from 1, got {self.value_reference} for {self.name!r}")
            pass
        if self.causality not in CAUSALITIES:
            raise ValueError(f"invalid causality {self.causality!r} for {self.name!r}, allowed: {CAUSALITIES}")
            pass
        if self.variability not in VARIABILITIES:
            raise ValueError(f"invalid variability {self.variability!r} for {self.name!r}, allowed: {VARIABILITIES}")
            pass
        if self.type not in TYPES:
            raise ValueError(f"invalid type {self.type!r} for {self.name!r}, allowed: {TYPES}")
            pass
        if self.initial is not None and self.initial not in INITIALS:
            raise ValueError(f"invalid initial {self.initial!r} for {self.name!r}, allowed: {INITIALS}")
            pass
        # * A parameter whose value is fixed after initialization must not be
        # * declared continuous; the FMI semantic rules reject that combination.
        if self.causality == "parameter" and self.variability == "continuous":
            raise ValueError(
                f"parameter {self.name!r} must be fixed or tunable, got variability='continuous'"
            )
            pass
        if self.causality == "input" and self.initial is not None:
            raise ValueError(f"input {self.name!r} must not declare initial")
            pass
        if self.causality == "independent" and (self.initial is not None or self.start is not None):
            raise ValueError(f"independent variable {self.name!r} must not declare initial or start")
            pass
        if self.initial in ("exact", "approx") and self.start is None:
            raise ValueError(f"variable {self.name!r} with initial={self.initial!r} requires start")
            pass
        if self.start is not None:
            format_start(self.start, self.type)
            pass
        pass
    pass


def build_variables(
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    require_initial_values: bool = False,
    initial_names: Sequence[str] | None = None,
    initial_values: Sequence[float] | None = None,
    include_time: bool = False,
) -> List[FMIVariable]:
    '''
    Allocates value references for the whole FMU:

        [1, X]                inputs   u(t)
        [X + 1, X + Y]        outputs  y(t)
        [X + Y + 1, X + 2Y]   initial outputs y(0), read once during initialization

    Names and state expansion are supplied by the pipeline's export spec.
    require_initial_values alone decides whether the initial-output block is
    present. With that flag enabled, an empty initial_values list gives every
    output a zero start value; a nonempty list must match the output count.
    initial_names must name every initial-output parameter when required.
    FMI 3.0 requires an independent time variable. Set include_time=True for
    that version; its value reference follows all backend-owned references.
    '''
    resolved_inputs: List[str] = list(input_names)
    resolved_outputs: List[str] = list(output_names)
    resolved_initial_names: List[str] = [] if initial_names is None else list(initial_names)
    if not resolved_inputs or not resolved_outputs:
        raise ValueError("input_names and output_names must not be empty")
    if any(not isinstance(name, str) or not name for name in resolved_inputs + resolved_outputs + resolved_initial_names):
        raise ValueError("variable names must be nonempty strings")

    provided_initials: List[float] = [] if initial_values is None else list(initial_values)
    resolved_initials: List[float] = []
    if require_initial_values:
        if len(resolved_initial_names) != len(resolved_outputs):
            raise ValueError(f"expected {len(resolved_outputs)} initial names, got {len(resolved_initial_names)}")
            pass
        if provided_initials and len(provided_initials) != len(resolved_outputs):
            raise ValueError(f"expected {len(resolved_outputs)} initial values, got {len(provided_initials)}")
            pass
        resolved_initials = [float(value) for value in provided_initials] if provided_initials else [0.0] * len(resolved_outputs)
        if any(not isfinite(value) for value in resolved_initials):
            raise ValueError("initial values must be finite")
            pass
        pass
    elif resolved_initial_names or provided_initials:
        raise ValueError("initial names or values were provided, but require_initial_values is False")
        pass

    variables: List[FMIVariable] = []
    value_reference: int = 1
    for name in resolved_inputs:
        variables.append(FMIVariable(
            name=name,
            value_reference=value_reference,
            causality="input",
            variability="continuous",
            start=0.0,
        ))
        value_reference += 1
        pass
    for name in resolved_outputs:
        variables.append(FMIVariable(
            name=name,
            value_reference=value_reference,
            causality="output",
            variability="continuous",
            initial="calculated",
        ))
        value_reference += 1
        pass
    for name, value in zip(resolved_initial_names, resolved_initials):
        variables.append(FMIVariable(
            name=name,
            value_reference=value_reference,
            causality="parameter",
            variability="fixed",
            initial="exact",
            start=value,
        ))
        value_reference += 1
        pass
    if include_time:
        variables.append(FMIVariable(
            name="time",
            value_reference=value_reference,
            causality="independent",
            variability="continuous",
        ))
        pass

    names: List[str] = [variable.name for variable in variables]
    seen: set[str] = set()
    duplicated: set[str] = set()
    for name in names:
        if name in seen:
            duplicated.add(name)
            pass
        seen.add(name)
        pass
    if duplicated:
        raise ValueError(f"duplicated variable names: {sorted(duplicated)}")
        pass
    return variables
    pass
