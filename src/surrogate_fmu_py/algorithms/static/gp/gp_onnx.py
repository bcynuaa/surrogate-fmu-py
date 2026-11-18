from __future__ import annotations

from operator import irshift
from typing import Any

import numpy as np


def _decompose_kernel(kernel: Any) -> dict[str, Any]:
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        Product,
        RBF,
        Sum,
        WhiteKernel,
    )

    found: dict[str, Any] = {
        "constant": 1.0,
        "kind": None,
        "length_scale": None,
        "nu": None,
    }

    def walk(node: Any) -> None:
        if isinstance(node, Sum):
            walk(node.k1)
            walk(node.k2)
            return
        if isinstance(node, Product):
            walk(node.k1)
            walk(node.k2)
            return
        if isinstance(node, ConstantKernel):
            found["constant"] *= float(node.constant_value)
            return
        if isinstance(node, WhiteKernel):
            return
        if isinstance(node, Matern):
            found["kind"] = "Matern"
            found["length_scale"] = np.atleast_1d(
                np.asarray(node.length_scale, dtype=np.float64)
            )
            found["nu"] = float(node.nu)
            return
        if isinstance(node, RBF):
            found["kind"] = "RBF"
            found["length_scale"] = np.atleast_1d(
                np.asarray(node.length_scale, dtype=np.float64)
            )
            found["nu"] = None
            return
        raise ValueError(
            f"kernel {type(node).__name__} is not supported by the GP ONNX exporter. "
            "Supported kernels are ConstantKernel * (RBF | Matern) + WhiteKernel."
        )
        pass

    walk(kernel)
    if found["kind"] is None or found["length_scale"] is None:
        raise ValueError("could not find an RBF/Matern component in the GP kernel")
        pass
    return found
    pass


def build_onnx(
    core: Any,
    input_name: str,
    output_name: str,
    input_dim: int,
) -> Any:
    import onnx
    from onnx import TensorProto, helper

    model = getattr(core, "model_", None)
    if model is None or not hasattr(model, "estimators_"):
        raise RuntimeError("GPCore is not fitted: call init() then fit()")
        pass

    estimators = list(model.estimators_)
    if len(estimators) == 0:
        raise RuntimeError("GPCore has no fitted estimators")
        pass

    first = estimators[0]
    train_x = np.asarray(first.X_train_, dtype=np.float64)
    n_train, model_dim = train_x.shape
    if model_dim != input_dim:
        raise ValueError(
            f"GP was fitted on {model_dim} features but the encoder expects {input_dim}"
        )
        pass

    nodes: list[Any] = []
    initializers: list[Any] = []

    def const(name: str, array: np.ndarray) -> str:
        shaped = np.asarray(array, dtype=np.float64)
        initializers.append(
            helper.make_tensor(
                name,
                TensorProto.DOUBLE,
                list(shaped.shape),
                shaped.ravel(),
            )
        )
        return name
        pass

    const("one", np.array([1.0], dtype=np.float64))
    const("train_x", train_x.reshape(1, n_train, model_dim))
    nodes.append(helper.make_node("Unsqueeze", [input_name], ["x_unsq"], axes=[1]))

    outputs: list[str] = []
    for index, estimator in enumerate(estimators):
        components = _decompose_kernel(estimator.kernel_)
        length_scale = np.atleast_1d(
            np.asarray(components["length_scale"], dtype=np.float64)
        )
        if length_scale.size == 1:
            length_scale = np.full((model_dim,), float(length_scale[0]))
        elif length_scale.size != model_dim:
            raise ValueError(
                f"length_scale has {length_scale.size} entries but the model has "
                f"{model_dim} features"
            )
            pass

        prefix = f"g{index}"
        nodes.append(helper.make_node("Sub", ["x_unsq", "train_x"], [f"{prefix}_diff"]))
        const(
            f"{prefix}_inv_l",
            (1.0 / length_scale).astype(np.float64).reshape(1, 1, model_dim),
        )
        nodes.append(
            helper.make_node(
                "Mul",
                [f"{prefix}_diff", f"{prefix}_inv_l"],
                [f"{prefix}_scaled"],
            )
        )
        nodes.append(
            helper.make_node(
                "Mul",
                [f"{prefix}_scaled", f"{prefix}_scaled"],
                [f"{prefix}_sq"],
            )
        )
        nodes.append(
            helper.make_node(
                "ReduceSum",
                [f"{prefix}_sq"],
                [f"{prefix}_d2"],
                axes=[2],
                keepdims=0,
            )
        )

        if components["kind"] == "RBF":
            halve = const(f"{prefix}_half", np.array([0.5], dtype=np.float64))
            nodes.append(
                helper.make_node(
                    "Mul",
                    [f"{prefix}_d2", halve],
                    [f"{prefix}_half_d2"],
                )
            )
            nodes.append(
                helper.make_node(
                    "Neg",
                    [f"{prefix}_half_d2"],
                    [f"{prefix}_exp_arg"],
                )
            )
            nodes.append(
                helper.make_node(
                    "Exp",
                    [f"{prefix}_exp_arg"],
                    [f"{prefix}_base"],
                )
            )
        else:
            nu = components["nu"]
            nodes.append(helper.make_node("Sqrt", [f"{prefix}_d2"], [f"{prefix}_r"]))
            if nu == 0.5:
                nodes.append(helper.make_node("Neg", [f"{prefix}_r"], [f"{prefix}_neg"]))
                nodes.append(
                    helper.make_node("Exp", [f"{prefix}_neg"], [f"{prefix}_base"])
                )
            elif nu == 1.5:
                k3 = const(f"{prefix}_k3", np.array([np.sqrt(3.0)], dtype=np.float64))
                nodes.append(helper.make_node("Mul", [f"{prefix}_r", k3], [f"{prefix}_u"]))
                nodes.append(
                    helper.make_node("Add", [f"{prefix}_u", "one"], [f"{prefix}_p1"])
                )
                nodes.append(helper.make_node("Neg", [f"{prefix}_u"], [f"{prefix}_neg"]))
                nodes.append(
                    helper.make_node("Exp", [f"{prefix}_neg"], [f"{prefix}_expo"])
                )
                nodes.append(
                    helper.make_node("Mul", [f"{prefix}_p1", f"{prefix}_expo"], [f"{prefix}_base"])
                )
            elif nu == 2.5:
                k5 = const(f"{prefix}_k5", np.array([np.sqrt(5.0)], dtype=np.float64))
                nodes.append(helper.make_node("Mul", [f"{prefix}_r", k5], [f"{prefix}_u"]))
                nodes.append(
                    helper.make_node("Mul", [f"{prefix}_u", f"{prefix}_u"], [f"{prefix}_u2"])
                )
                one_third = const(f"{prefix}_one_third", np.array([1.0 / 3.0], dtype=np.float64))
                nodes.append(
                    helper.make_node(
                        "Mul",
                        [f"{prefix}_u2", one_third],
                        [f"{prefix}_t2"],
                    )
                )
                nodes.append(
                    helper.make_node("Add", [f"{prefix}_u", f"{prefix}_t2"], [f"{prefix}_t01"])
                )
                nodes.append(
                    helper.make_node("Add", [f"{prefix}_t01", "one"], [f"{prefix}_poly"])
                )
                nodes.append(helper.make_node("Neg", [f"{prefix}_u"], [f"{prefix}_neg"]))
                nodes.append(
                    helper.make_node("Exp", [f"{prefix}_neg"], [f"{prefix}_expo"])
                )
                nodes.append(
                    helper.make_node("Mul", [f"{prefix}_poly", f"{prefix}_expo"], [f"{prefix}_base"])
                )
            else:
                raise ValueError(
                    f"unsupported Matern nu={nu!r}; supported values are 0.5, 1.5, and 2.5"
                )
                pass

        kernel_constant = const(
            f"{prefix}_c",
            np.array([components["constant"]], dtype=np.float64),
        )
        nodes.append(
            helper.make_node(
                "Mul",
                [f"{prefix}_base", kernel_constant],
                [f"{prefix}_k"],
            )
        )

        alpha = np.asarray(estimator.alpha_, dtype=np.float64).reshape(n_train, 1)
        raw_mean = np.asarray(
            getattr(estimator, "_y_train_mean", np.array([0.0], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        mean = float(raw_mean[0]) if raw_mean.size else 0.0
        const(f"{prefix}_alpha", alpha)
        const(f"{prefix}_mean", np.array([mean], dtype=np.float64))
        nodes.append(
            helper.make_node(
                "MatMul",
                [f"{prefix}_k", f"{prefix}_alpha"],
                [f"{prefix}_raw"],
            )
        )
        nodes.append(
            helper.make_node(
                "Add",
                [f"{prefix}_raw", f"{prefix}_mean"],
                [f"{prefix}_out"],
            )
        )
        outputs.append(f"{prefix}_out")

    if len(outputs) == 1:
        nodes.append(helper.make_node("Identity", [outputs[0]], [output_name]))
    else:
        nodes.append(helper.make_node("Concat", outputs, [output_name], axis=1))

    graph = helper.make_graph(
        nodes,
        "gp_analytic",
        [helper.make_tensor_value_info(input_name, TensorProto.DOUBLE, [None, input_dim])],
        [helper.make_tensor_value_info(output_name, TensorProto.DOUBLE, [None, len(outputs)])],
        initializer=initializers,
    )
    # * currently fix ir_version to 9 for compatibility with onnxruntime
    proto = helper.make_model(graph, producer_name="surrogate-fmu", ir_version=9)
    proto.opset_import[0].version = 11
    onnx.checker.check_model(proto)
    return proto
    pass
