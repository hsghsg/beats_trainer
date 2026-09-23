"""读取并严格校验集中式推理配置。"""

import math
from pathlib import Path
from urllib.parse import urlparse

import yaml

BACKENDS = {
    "beats-native",
    "local-density-knn",
    "relative-mahalanobis",
    "gmm-cosine-knn",
}


def require_number(value, name: str, minimum: float, integer: bool = False) -> None:
    """校验配置为有限数值且不小于下限；integer 为真时只允许 int。"""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (integer and type(value) is not int)
    ):
        raise ValueError(
            f"{name} 必须是至少为 {minimum} 的有限{'整数' if integer else '数值'}"
        )


def resolve_path(base: Path, value) -> Path:
    """将非空路径按 YAML 所在目录解析，支持绝对路径与用户目录。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("配置路径必须是非空字符串")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(filename: str | Path) -> dict:
    """读取 infer.yaml，校验参数及投票条件并解析路径；错误抛出 ValueError。

    配置项不使用隐式合并，缺失字段须在 YAML 中明确补全；所有相对输入、
    模型和输出路径均以配置文件所在目录为基准，与工作目录无关。
    """
    filename = Path(filename).expanduser().resolve()
    with filename.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    try:
        validate_config(config)
        audio = config["input"]
        audio["path"] = resolve_path(filename.parent, audio["path"])
        config["embedding"]["checkpoint"] = resolve_path(
            filename.parent, config["embedding"]["checkpoint"]
        )
        for name, backend in config["classifiers"].items():
            if backend["enabled"]:
                key = "checkpoint" if name == "beats-native" else "path"
                backend[key] = resolve_path(filename.parent, backend[key])
        for key in ("jsonl_path", "log_path"):
            if config["output"][key] is not None:
                config["output"][key] = resolve_path(
                    filename.parent, config["output"][key]
                )
    except (KeyError, TypeError) as error:
        raise ValueError(f"infer.yaml 配置缺失或类型错误：{error}") from error
    return config


def validate_config(config: dict) -> None:
    """校验输入、前处理、后端阈值和输出配置，禁止静默忽略非法模式。"""
    if not isinstance(config, dict):
        raise TypeError("infer.yaml 必须为配置映射")
    audio, front = config["input"], config["frontend"]
    embedding, ensemble = config["embedding"], config["ensemble"]
    if audio["mode"] not in {"file", "websocket"}:
        raise ValueError("input.mode 只支持 file 或 websocket")
    if audio["sample_rate"] != 16000 or audio["duration_seconds"] != 10:
        raise ValueError("当前 BEATs 推理固定为 16000 Hz、10 秒窗口")
    if audio["tail_policy"] not in {"pad", "drop", "error"}:
        raise ValueError("tail_policy 只支持 pad、drop 或 error")
    if type(audio["recursive"]) is not bool or type(embedding["normalize"]) is not bool:
        raise ValueError("recursive 和 normalize 必须使用 YAML 布尔值")
    if front["method"] not in {"log-mel-energy", "wavelet"}:
        raise ValueError("前处理只支持 log-mel-energy 或 wavelet")
    require_number(front["fbank_std"], "fbank_std", 1e-12)
    require_number(front["fbank_mean"], "fbank_mean", -1e30)
    for key in ("cwt_voices_per_octave", "cwt_time_stride", "max_tokens"):
        require_number(front[key], key, 1, integer=True)
    require_number(embedding["num_threads"], "num_threads", 1, integer=True)
    if not isinstance(embedding["backbone_overrides"], dict):
        raise TypeError("backbone_overrides 必须为参数映射")
    if not isinstance(embedding["device"], str):
        raise TypeError("device 必须是设备名称字符串")
    labels = [ensemble["normal_label"], ensemble["abnormal_label"]]
    if (
        any(not isinstance(label, str) or not label for label in labels)
        or labels[0] == labels[1]
    ):
        raise ValueError("正常与异常标签必须为不同的非空字符串")
    backends = config["classifiers"]
    if not isinstance(backends, dict) or set(backends) - BACKENDS:
        raise ValueError("配置包含不支持的分类器名称")
    enabled = 0
    for name, backend in backends.items():
        if type(backend["enabled"]) is not bool:
            raise ValueError(f"{name}.enabled 必须使用 YAML 布尔值")
        if not backend["enabled"]:
            continue
        enabled += 1
        require_number(backend["threshold"], f"{name}.threshold", 0)
        if backend["threshold"] > 1:
            raise ValueError(f"{name} 的阈值不能大于 1")
        if name == "beats-native" and (
            not isinstance(backend["labels"], list)
            or len(backend["labels"]) != 2
            or set(backend["labels"]) != set(labels)
        ):
            raise ValueError("原生分类头 labels 必须恰好包含配置的正常与异常标签")
    require_number(ensemble["min_anomalous"], "min_anomalous", 1, integer=True)
    if ensemble["min_anomalous"] > enabled:
        raise ValueError("m 不得超过启用的分类器数量，且至少启用一个分类器")
    ws = audio["websocket"]
    address = urlparse(ws["url"])
    if address.scheme not in {"ws", "wss"} or not address.hostname:
        raise ValueError("WebSocket 地址必须使用 ws:// 或 wss://")
    if ws["format"] not in {"s16le", "f32le"}:
        raise ValueError("PCM 格式只支持 s16le 和 f32le")
    for key in ("sample_rate", "channels", "max_message_bytes"):
        require_number(ws[key], key, 1, integer=True)
    require_number(ws["max_windows"], "max_windows", 0, integer=True)
    require_number(ws["timeout_seconds"], "timeout_seconds", 0.01)
    if ws["request"] is not None and not isinstance(ws["request"], str):
        raise ValueError("WebSocket request 必须是文本或 null")
    if type(ws["request_each_window"]) is not bool:
        raise ValueError("request_each_window 必须为布尔值")
    if ws["request_each_window"] and ws["request"] is None:
        raise ValueError("重复请求模式必须配置 request 文本")
    if config["output"]["log_level"] not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("不支持的日志级别")
