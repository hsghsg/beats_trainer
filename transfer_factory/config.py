"""转换配置解析、路径处理及参数校验。"""

import math
import re
from pathlib import Path

import yaml


def resolve_path(base: Path, value: str) -> Path:
    """将非空路径按配置目录解析为绝对路径，拒绝在 Linux 使用 Windows 盘符。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("路径必须是非空字符串")
    if re.match(r"^[A-Za-z]:[\\/]", value) and base.as_posix().startswith("/"):
        raise ValueError(
            "WSL 配置请使用 /mnt/c/... 等 Linux 路径，不要使用 Windows 盘符"
        )
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def positive_integer(value, name: str, minimum: int = 1) -> None:
    """校验整数参数及其下限，排除布尔值和小数。"""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须是至少为 {minimum} 的整数")


def load_config(filename: str | Path) -> dict:
    """加载 YAML，校验输入输出契约，并按文件所在目录解析全部模型/日志路径。

    不检查模型存在性和 CANN 环境，允许独立使用 ONNX→OM 阶段及 doctor。
    缺失字段、非法 shape、重复输出名等均抛出 ValueError。
    """
    filename = Path(filename).expanduser().resolve()
    with filename.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    try:
        model, inp, onnx, atc, runtime = (
            config[key] for key in ("model", "input", "onnx", "atc", "runtime")
        )
        if model["kind"] not in {"beats", "torchscript", "python_factory"}:
            raise ValueError("model.kind 只支持 beats、torchscript、python_factory")
        if not isinstance(inp["shape"], list) or not inp["shape"]:
            raise ValueError("input.shape 必须为非空维度列表")
        for dimension in inp["shape"]:
            positive_integer(dimension, "input.shape 维度")
        if inp["dtype"] != "float32":
            raise ValueError("当前转换只支持 float32 输入")
        if not isinstance(inp["name"], str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", inp["name"]
        ):
            raise ValueError("input.name 必须为合法张量名称")
        names = onnx["output_names"]
        if (
            not isinstance(names, list)
            or not names
            or len(names) != len(set(names))
            or any(
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                for name in names
            )
            or inp["name"] in names
        ):
            raise ValueError("ONNX 输出名须非空、合法、唯一，且不能与输入重名")
        positive_integer(onnx["opset"], "onnx.opset", 13)
        positive_integer(onnx["validation_runs"], "validation_runs")
        for key in ("atol", "rtol"):
            value = onnx[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{key} 必须为有限非负数")
        for value in (
            onnx["verify"],
            runtime["overwrite"],
            model["normalize_embedding"],
        ):
            if type(value) is not bool:
                raise ValueError(
                    "verify、overwrite、normalize_embedding 必须是 YAML 布尔值"
                )
        positive_integer(runtime["num_threads"], "num_threads")
        positive_integer(runtime["seed"], "seed", 0)
        positive_integer(atc["timeout_seconds"], "timeout_seconds")
        if not isinstance(atc["extra_args"], dict) or not isinstance(
            model["backbone_overrides"], dict
        ):
            raise TypeError("extra_args 和 backbone_overrides 必须为参数映射")
        if model["kind"] == "beats":
            modes = {
                "embedding": ["embedding"],
                "probabilities": ["probabilities"],
                "both": ["embedding", "probabilities"],
            }
            if (
                model["output_mode"] not in modes
                or names != modes[model["output_mode"]]
            ):
                raise ValueError("BEATs output_mode 与 output_names 不匹配")
            if len(inp["shape"]) != 3:
                raise ValueError("BEATs 输入须为 [batch,time,freq] 时频特征")
            for key in ("fbank_mean", "fbank_std"):
                value = model[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{key} 必须为有限数值")
            if model["fbank_std"] <= 0:
                raise ValueError("fbank_std 必须大于零")
            positive_integer(model["max_tokens"], "max_tokens")
        for section, key in (
            (model, "checkpoint"),
            (onnx, "path"),
            (atc, "output"),
            (runtime, "report_path"),
            (runtime, "log_path"),
        ):
            section[key] = resolve_path(filename.parent, section[key])
        atc.setdefault("devlib_path", None)
        for section, key in (
            (inp, "sample_npy"),
            (atc, "env_script"),
            (atc, "devlib_path"),
        ):
            if section[key] is not None:
                section[key] = resolve_path(filename.parent, section[key])
        if atc["output"].suffix == ".om":
            raise ValueError("atc.output 是前缀，请去掉 .om 扩展名")
        protected = {
            model["checkpoint"],
            onnx["path"],
            Path(str(atc["output"]) + ".om"),
            runtime["report_path"],
            runtime["log_path"],
        }
        if (
            len(protected) != 5
            or filename in protected
            or inp["sample_npy"] in protected
        ):
            raise ValueError("模型、产物、配置、样本、报告和日志路径不能相互覆盖")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"转换配置缺失或类型错误：{error}") from error
    return config
