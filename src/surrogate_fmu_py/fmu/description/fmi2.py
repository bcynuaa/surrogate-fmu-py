from __future__ import annotations

from typing import List
from xml.etree import ElementTree

from .variable import FMIVariable
from .utils import format_number, format_start


# * FMI 2.0 spells the interface element and the typed element differently from
# * FMI 3.0. Type names in FMIVariable follow FMI 3.0, so map them back here.
FMI_INTERFACE_ELEMENTS: dict[str, str] = {
    "cs": "CoSimulation",
    "me": "ModelExchange",
}
TYPE_ELEMENTS: dict[str, str] = {
    "Float64": "Real",
    "Int32": "Integer",
    "Boolean": "Boolean",
    "String": "String",
}


def render(model) -> bytes:
    root: ElementTree.Element = ElementTree.Element(
        "fmiModelDescription",
        {
            "fmiVersion": "2.0",
            "modelName": model.model_name,
            "guid": model.token,
            "generationTool": model.generation_tool,
            "variableNamingConvention": model.variable_naming_convention,
            **({"description": model.description} if model.description else {}),
        },
    )
    append_fmi_interface(root, model, model.fmi_interface)
    if model.step_size is not None:
        ElementTree.SubElement(root, "DefaultExperiment", {"stepSize": format_number(model.step_size)})
        pass

    variables_element: ElementTree.Element = ElementTree.SubElement(root, "ModelVariables")
    for variable in model.variables:
        append_scalar_variable(variables_element, variable)
        pass

    append_model_structure(root, model)

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
    pass


def append_fmi_interface(root: ElementTree.Element, model, fmi_interface: str) -> None:
    if fmi_interface == "me":
        # * The C++ side has no model exchange entry point yet, so emitting the
        # * element would describe a binary that cannot serve it. ModelExchange
        # * also needs numberOfEventIndicators and a Derivatives block.
        raise NotImplementedError("FMI 2.0 ModelExchange is not implemented yet")
        pass
    ElementTree.SubElement(
        root,
        FMI_INTERFACE_ELEMENTS[fmi_interface],
        {
            "modelIdentifier": model.model_identifier,
            "canHandleVariableCommunicationStepSize": str(
                model.can_handle_variable_communication_step_size
            ).lower(),
            "canGetAndSetFMUstate": "false",
            "canSerializeFMUstate": "false",
        },
    )
    pass


def append_scalar_variable(parent: ElementTree.Element, variable: FMIVariable) -> None:
    if variable.causality == "structuralParameter":
        raise NotImplementedError("FMI 2.0 does not support structuralParameter")
        pass
    if variable.type not in TYPE_ELEMENTS:
        raise NotImplementedError(f"FMI 2.0 cannot express type {variable.type!r} ({variable.name!r})")
        pass
    attributes: dict = {
        "name": variable.name,
        "valueReference": str(variable.value_reference),
        "causality": variable.causality,
        "variability": variable.variability,
    }
    if variable.initial is not None:
        attributes["initial"] = variable.initial
        pass
    if variable.description:
        attributes["description"] = variable.description
        pass
    element: ElementTree.Element = ElementTree.SubElement(parent, "ScalarVariable", attributes)
    # * FMI 2.0 keeps start inside a typed child element, FMI 3.0 moves it up to
    # * an attribute of the typed element itself.
    typed: ElementTree.Element = ElementTree.SubElement(element, TYPE_ELEMENTS[variable.type])
    if variable.start is not None:
        typed.set("start", format_start(variable.start, variable.type))
        pass
    pass


def append_model_structure(root: ElementTree.Element, model) -> None:
    structure_element: ElementTree.Element = ElementTree.SubElement(root, "ModelStructure")
    # * Outputs carry no dependencies on purpose: the attribute describes the
    # * dependency on the Knowns and an absent attribute already means "depends on
    # * all of them", which is exactly what a surrogate network does. Writing an
    # * empty list instead would claim the outputs are constant.
    append_unknowns(structure_element, "Outputs", model, "output")
    # * During initialization, window and NeuralODE outputs depend on start
    # * parameters; vector_coder_onnx outputs depend on the inputs.
    append_unknowns(structure_element, "InitialUnknowns", model, "output", initial_dependencies(model))
    pass


def initial_dependencies(model) -> List[str]:
    '''
    Positions (1-based, ModelVariables order) that drive the outputs during
    initialization. FMI 2.0 numbers dependencies the same way it numbers
    Unknown/@index, not by valueReference.
    '''
    sources = model.initial_dependencies()
    positions: dict[int, int] = {
        variable.value_reference: index
        for index, variable in enumerate(model.variables, start=1)
    }
    return [str(positions[variable.value_reference]) for variable in sources]
    pass


# * FMI 2.0 refers to Unknowns by their 1-based position in ModelVariables, not
# * by valueReference. The two coincide only while the references stay dense.
def append_unknowns(
    parent: ElementTree.Element,
    tag: str,
    model,
    causality: str,
    dependencies: List[str] | None = None,
) -> None:
    unknowns: ElementTree.Element = ElementTree.SubElement(parent, tag)
    for index, variable in enumerate(model.variables, start=1):
        if variable.causality != causality:
            continue
            pass
        attributes: dict = {"index": str(index)}
        if dependencies:
            attributes["dependencies"] = " ".join(dependencies)
            attributes["dependenciesKind"] = " ".join(["dependent"] * len(dependencies))
            pass
        ElementTree.SubElement(unknowns, "Unknown", attributes)
        pass
    pass
