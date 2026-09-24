"""PyTorch 到 ONNX 导出、静态模型契约检查及数值一致性验证。"""

import hashlib
import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """分块计算模型文件 SHA-256，避免将大模型一次性读取到内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_onnx_contract(path: Path, config: dict) -> dict:
    """校验 ONNX 结构及单输入 float32 静态 shape，返回可供 ATC 和报告使用的元信息。

    输入名称、维度和输出名称必须与配置一致，防止 ATC 按错误输入契约编译。
    本版仅接受标准 ONNX 算子及内嵌权重模型。
    """
    graph = onnx.load(str(path), load_external_data=False)
    if any(
        tensor.data_location == TensorProto.EXTERNAL
        for tensor in graph.graph.initializer
    ):
        raise ValueError("当前工厂仅支持内嵌权重 ONNX，请先合并 external data")
    onnx.checker.check_model(graph)
    if any(node.domain not in ("", "ai.onnx") for node in graph.graph.node):
        raise ValueError("ONNX 含自定义算子域，需要额外 ATC 算子适配，不能直接转换")
    initializers = {value.name for value in graph.graph.initializer}
    inputs = [value for value in graph.graph.input if value.name not in initializers]
    if len(inputs) != 1:
        raise ValueError("当前工厂仅支持单输入模型")
    tensor = inputs[0].type.tensor_type
    shape = [dimension.dim_value for dimension in tensor.shape.dim]
    inp = config["input"]
    if (
        inputs[0].name != inp["name"]
        or shape != inp["shape"]
        or tensor.elem_type != TensorProto.FLOAT
    ):
        raise ValueError(
            f"ONNX 输入契约不匹配：实际 {inputs[0].name}:{shape}，需要 {inp['name']}:{inp['shape']}"
        )
    outputs = [value.name for value in graph.graph.output]
    if outputs != config["onnx"]["output_names"]:
        raise ValueError(f"ONNX 输出名称与配置不一致：{outputs}")
    return {
        "input_name": inputs[0].name,
        "input_shape": shape,
        "input_dtype": "float32",
        "output_names": outputs,
        "opsets": {item.domain: item.version for item in graph.opset_import},
        "operators": sorted({node.op_type for node in graph.graph.node}),
    }


def sample_input(config: dict, rng: np.random.Generator, use_file: bool) -> np.ndarray:
    """加载精确 shape/dtype 的真实 .npy，或生成确定性 float32 随机验证输入。"""
    filename = config["input"]["sample_npy"]
    shape = tuple(config["input"]["shape"])
    if use_file and filename is not None:
        value = np.load(filename, allow_pickle=False)
    else:
        value = rng.standard_normal(shape).astype(np.float32)
    if (
        value.shape != shape
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("样本输入的 shape、float32 类型或有限性不满足配置")
    return np.ascontiguousarray(value)


def tensor_outputs(value) -> tuple:
    """将单个张量或扁平张量序列统一为元组，拒绝字典和嵌套输出。"""
    import torch

    values = (value,) if isinstance(value, torch.Tensor) else value
    if (
        not isinstance(values, (tuple, list))
        or not values
        or any(not isinstance(item, torch.Tensor) for item in values)
    ):
        raise TypeError("模型输出必须为张量或非空的扁平张量序列")
    if any(not torch.isfinite(item).all() for item in values):
        raise ValueError("PyTorch 模型输出包含非有限值")
    return tuple(values)


def export_onnx(config: dict) -> dict:
    """导出固定 shape 的标准 ONNX，结构与 ORT 数值检查成功后原子替换目标文件。

    导出或验证失败时不会覆盖已有成功模型。导出仅使用 CPU、eval 模式和
    torch.onnx 的经典导出器，避免引入 ATen fallback 自定义算子。
    """
    import torch

    from .adapters import load_model

    settings = config["onnx"]
    output = settings["path"]
    if output.exists() and not config["runtime"]["overwrite"]:
        raise FileExistsError(f"ONNX 已存在，请配置 overwrite 或更换输出路径：{output}")
    torch.set_num_threads(config["runtime"]["num_threads"])
    torch.manual_seed(config["runtime"]["seed"])
    rng = np.random.default_rng(config["runtime"]["seed"])
    network = load_model(config)
    sample = sample_input(config, rng, use_file=True)
    example = torch.from_numpy(sample)
    with torch.inference_mode():
        expected = tensor_outputs(network(example))
    if len(expected) != len(settings["output_names"]):
        raise ValueError("实际模型输出数量与 output_names 不一致")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".onnx-", dir=output.parent) as directory:
        temporary = Path(directory) / "model.onnx"
        LOGGER.info("开始导出 ONNX：%s", output)
        with torch.no_grad():
            torch.onnx.export(
                network,
                example,
                str(temporary),
                export_params=True,
                opset_version=settings["opset"],
                do_constant_folding=True,
                input_names=[config["input"]["name"]],
                output_names=settings["output_names"],
                dynamo=False,
            )
        metadata = check_onnx_contract(temporary, config)
        checks = []
        if settings["verify"]:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = config["runtime"]["num_threads"]
            session = ort.InferenceSession(
                str(temporary), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for run in range(settings["validation_runs"]):
                data = sample if run == 0 else sample_input(config, rng, use_file=False)
                with torch.inference_mode():
                    reference = tensor_outputs(network(torch.from_numpy(data)))
                actual = session.run(
                    settings["output_names"], {config["input"]["name"]: data}
                )
                for name, before, after in zip(
                    settings["output_names"], reference, actual
                ):
                    before = before.detach().cpu().numpy()
                    if before.shape != after.shape or not np.isfinite(after).all():
                        raise ValueError(f"ONNX 输出 {name} 形状或有限性校验失败")
                    np.testing.assert_allclose(
                        after,
                        before,
                        atol=settings["atol"],
                        rtol=settings["rtol"],
                        err_msg=f"ONNX 输出 {name} 与 PyTorch 不一致",
                    )
                    checks.append(
                        {
                            "run": run,
                            "output": name,
                            "max_abs_error": float(np.max(np.abs(after - before))),
                        }
                    )
            del session
        if output.exists() and not config["runtime"]["overwrite"]:
            raise FileExistsError(f"导出期间目标文件被创建：{output}")
        temporary.replace(output)
    LOGGER.info("ONNX 导出成功：%s，数值验证=%s", output, settings["verify"])
    return {
        "stage": "pt-to-onnx",
        "status": "success",
        "output": str(output),
        "sha256": sha256_file(output),
        "contract": metadata,
        "verified": settings["verify"],
        "checks": checks,
        "model_kind": config["model"]["kind"],
        "labels": config["model"]["labels"],
        "torch_version": torch.__version__,
    }
