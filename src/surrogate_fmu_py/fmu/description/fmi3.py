from __future__ import annotations

from typing import List
from xml.etree import ElementTree

from .variable import FMIVariable
from .utils import format_number, format_start


# * FMI 3.0 names the interface element and the typed element, and unlike 2.0 the
# * typed element *is* the variable: there is no ScalarVariable wrapper and start
# * is an attribute of that element.
FMI_INTERFACE_ELEMENTS: dict[str, str] = {
    "cs": "CoSimulation",
    "me": "ModelExchange",
}
TYPE_ELEMENTS: dict[str, str] = {
    "Float64": "Float64",
    "Float32": "Float32",
    "Int32": "Int32",
    "Int64": "Int64",
    "Boolean": "Boolean",
    "String": "String",
}


def render(model) -> bytes:
    root: ElementTree.Element = ElementTree.Element(
        "fmiModelDescription",
        {
            "fmiVersion": "3.0",
            "modelName": model.model_name,
            # * FMI 3.0 replaced guid by instantiationToken. Same intent, and it
            # * is required, so derive it from the payload the same way.
            "instantiationToken": model.token,
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
        append_variable(variables_element, variable)
        pass

    append_model_structure(root, model)

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
    pass


def append_fmi_interface(root: ElementTree.Element, model, fmi_interface: str) -> None:
    if fmi_interface == "me":
        # * The C++ side has no FMI 3.0 model exchange entry point yet.
        # * ModelExchange would also need a <ContinuousStateDerivative> block.
        raise NotImplementedError("FMI 3.0 ModelExchange is not implemented yet")
        pass
    ElementTree.SubElement(
        root,
        FMI_INTERFACE_ELEMENTS[fmi_interface],
        {
            "modelIdentifier": model.model_identifier,
            "canHandleVariableCommunicationStepSize": str(
                model.can_handle_variable_communication_step_size
            ).lower(),
            # * The model never asks for an event iteration, so it does not
            # * implement the event mode.
            "hasEventMode": "false",
        },
    )
    pass


def append_variable(parent: ElementTree.Element, variable: FMIVariable) -> None:
    if variable.type not in TYPE_ELEMENTS:
        raise NotImplementedError(f"FMI 3.0 type {variable.type!r} is not supported ({variable.name!r})")
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
    if variable.start is not None:
        attributes["start"] = format_start(variable.start, variable.type)
        pass
    ElementTree.SubElement(parent, TYPE_ELEMENTS[variable.type], attributes)
    pass


def append_model_structure(root: ElementTree.Element, model) -> None:
    structure_element: ElementTree.Element = ElementTree.SubElement(root, "ModelStructure")
    # * Outputs carry no dependencies on purpose: an absent attribute already
    # * means "depends on all Knowns", which is what a surrogate network does.
    for variable in model.outputs():
        ElementTree.SubElement(structure_element, "Output", {"valueReference": str(variable.value_reference)})
        pass
    # * Initial outputs depend on start parameters for window and NeuralODE
    # * models, or on inputs for vector_coder_onnx.
    dependencies: List[str] = [str(variable.value_reference) for variable in model.initial_dependencies()]
    for variable in model.outputs():
        attributes: dict = {"valueReference": str(variable.value_reference)}
        if dependencies:
            attributes["dependencies"] = " ".join(dependencies)
            attributes["dependenciesKind"] = " ".join(["dependent"] * len(dependencies))
            pass
        ElementTree.SubElement(structure_element, "InitialUnknown", attributes)
        pass
    pass
