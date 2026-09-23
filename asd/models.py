"""离线加载 BEATs checkpoint 及项目的四种后端分类器。"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import compat
from .frontend import Frontend
from .vendor.BEATs.BEATs import BEATs, BEATsConfig
from .vendor.beats_trainer.classifiers import (
    GMMCosineKNNHybridClassifier,
    LocalDensityKNNClassifier,
    RelativeMahalanobisDistanceClassifier,
)

LOGGER = logging.getLogger(__name__)
CLASSIFIER_TYPES = {
    "local-density-knn": LocalDensityKNNClassifier,
    "relative-mahalanobis": RelativeMahalanobisDistanceClassifier,
    "gmm-cosine-knn": GMMCosineKNNHybridClassifier,
}


def as_mapping(value) -> dict:
    """将 checkpoint 中的配置字典或 dataclass 对象统一转换为字段映射。"""
    return value if isinstance(value, dict) else vars(value)


def build_backbone_config(
    checkpoint: dict, state: dict, overrides: dict
) -> BEATsConfig:
    """按项目默认值和权重形状重建网络配置，显式覆盖项用于特殊训练结构。

    Lightning 不保存完整 BEATs 架构参数，因此沿用原特征提取器默认值，
    从权重恢复维度/层数/相对位置配置，特殊非张量参数可在 YAML 覆盖。
    """
    cfg = {
        "activation_fn": "gelu",
        "attention_dropout": 0.1,
        "activation_dropout": 0.1,
        "encoder_layerdrop": 0.0,
        "dropout_input": 0.1,
        "layer_norm_first": False,
        "conv_bias": False,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "relative_position_embedding": True,
        "num_buckets": 320,
        "max_distance": 800,
        "gru_rel_pos": True,
        "deep_norm": True,
        "layer_wise_gradient_decay_ratio": 1.0,
    }
    if "state_dict" in checkpoint:
        saved = checkpoint.get("hyper_parameters", {}).get("config")
        if saved is None:
            raise ValueError("Lightning checkpoint 缺少 hyper_parameters.config")
        model = as_mapping(as_mapping(saved)["model"])
        for key in (
            "encoder_layers",
            "encoder_embed_dim",
            "encoder_ffn_embed_dim",
            "encoder_attention_heads",
            "input_patch_size",
            "embed_dim",
        ):
            if key in model:
                cfg[key] = model[key]
        cfg["dropout"] = model.get("dropout_rate", 0.1)
    else:
        cfg.update(as_mapping(checkpoint["cfg"]))
    patch = state["patch_embedding.weight"]
    cfg.update(
        embed_dim=patch.shape[0],
        input_patch_size=patch.shape[-1],
        conv_bias="patch_embedding.bias" in state,
        encoder_embed_dim=state["encoder.layers.0.self_attn.q_proj.weight"].shape[0],
        encoder_ffn_embed_dim=state["encoder.layers.0.fc1.weight"].shape[0],
    )
    cfg["encoder_layers"] = len(
        {key.split(".")[2] for key in state if key.startswith("encoder.layers.")}
    )
    bias_key = "encoder.layers.0.self_attn.relative_attention_bias.weight"
    cfg["relative_position_embedding"] = bias_key in state
    if bias_key in state:
        cfg["num_buckets"], cfg["encoder_attention_heads"] = state[bias_key].shape
    cfg["gru_rel_pos"] = any("grep_linear" in key for key in state)
    cfg.update(overrides)
    cfg["finetuned_model"] = False
    return BEATsConfig(cfg)


def build_native_head(checkpoint: dict, dimension: int) -> nn.Module:
    """按项目训练时的线性/MLP 结构构建原生分类头并严格加载参数。

    使用 checkpoint 保存的 activation、hidden_dims 和 dropout；原生头
    输入为未 L2 归一化的平均 embedding，输出为 softmax 前的 logits。
    """
    if "state_dict" not in checkpoint:
        raise ValueError("beats-native 需要项目训练的 Lightning .ckpt 分类头")
    params = checkpoint["hyper_parameters"]
    model = as_mapping(as_mapping(params["config"])["model"])
    state = {
        key.removeprefix("classifier."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("classifier.")
    }
    if not state:
        raise ValueError("checkpoint 未包含训练后的 classifier 分类头")
    num_classes = params.get("num_classes") or model.get("num_classes")
    if num_classes is None:
        raise ValueError("checkpoint 缺少原生分类头类别数")
    hidden = model.get("hidden_dims") if model.get("use_custom_head") else None
    if hidden:
        layers = []
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(dimension, width),
                    nn.ReLU()
                    if model.get("activation", "relu") == "relu"
                    else nn.GELU(),
                    nn.Dropout(model.get("dropout_rate", 0.1)),
                ]
            )
            dimension = width
        layers.append(nn.Linear(dimension, num_classes))
        head = nn.Sequential(*layers)
    else:
        head = nn.Linear(dimension, num_classes)
    head.load_state_dict(state, strict=True)
    return head


class BEATsModel:
    """仅包含推理网络，无 Lightning、训练数据或远程下载依赖。"""

    def __init__(
        self,
        path: Path,
        device: torch.device,
        overrides: dict,
        need_native: bool = False,
    ):
        """离线加载可信 checkpoint；缺失实际推理权重时报错，禁止随机权重推理。"""
        checkpoint = torch.load(
            path, map_location="cpu", pickle_module=compat, weights_only=False
        )
        if "state_dict" in checkpoint:
            state = {
                key.removeprefix("backbone."): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("backbone.")
            }
        else:
            state = {
                key: value
                for key, value in checkpoint["model"].items()
                if not key.startswith("predictor.")
            }
        cfg = build_backbone_config(checkpoint, state, overrides)
        self.backbone = BEATs(cfg)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        optional = {"patch_embedding_pad.weight", "raw2fbank_pad.weight"}
        if set(missing) - optional or unexpected:
            raise ValueError(
                f"BEATs 权重不完整或架构不匹配：缺失 {missing}，额外 {unexpected}"
            )
        self.head = (
            build_native_head(checkpoint, cfg.encoder_embed_dim)
            if need_native
            else None
        )
        self.backbone.to(device).eval()
        if self.head is not None:
            self.head.to(device).eval()
        self.device = device
        LOGGER.info("已加载 BEATs 模型：%s，设备：%s", path, device)

    def embedding(self, features: torch.Tensor, front: dict) -> torch.Tensor:
        """对未标准化的时频特征执行 BEATs，返回 [1,dim] 未归一化平均 embedding。"""
        encoded, _ = self.backbone.extract_features(
            features.to(self.device),
            fbank_mean=front["fbank_mean"],
            fbank_std=front["fbank_std"],
        )
        return encoded.mean(dim=1)


class BackendRunner:
    """共享 BEATs 前向计算并返回每个启用后端的异常分数。"""

    def __init__(self, config: dict):
        """检查权重路径并加载所需模型；原生和 embedding 可使用不同 checkpoint。"""
        self.config = config
        embedding = config["embedding"]
        torch.set_num_threads(embedding["num_threads"])
        device_name = embedding["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("部署仅支持 cpu 或 cuda 设备")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("指定了 CUDA，但当前环境没有可用的 CUDA")
        self.frontend = Frontend(config["frontend"])
        self.backends = {
            name: value
            for name, value in config["classifiers"].items()
            if value["enabled"]
        }
        self.estimators = {}
        self.models = {}
        self.classes = {}
        self.embedding_path = embedding["checkpoint"]
        native = self.backends.get("beats-native")
        required = (
            {self.embedding_path}
            if any(name != "beats-native" for name in self.backends)
            else set()
        )
        if native:
            required.add(native["checkpoint"])
        files = required | {
            value["path"]
            for name, value in self.backends.items()
            if name != "beats-native"
        }
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "推理权重文件不存在，请配置 infer.yaml：" + "、".join(sorted(missing))
            )
        for checkpoint in sorted(required):
            need_native = bool(native and native["checkpoint"] == checkpoint)
            self.models[checkpoint] = BEATsModel(
                checkpoint,
                self.device,
                embedding.get("backbone_overrides", {}),
                need_native,
            )
        labels = {
            config["ensemble"]["normal_label"],
            config["ensemble"]["abnormal_label"],
        }
        for name, backend in self.backends.items():
            if name == "beats-native":
                self.classes[name] = list(backend["labels"])
                head = self.models[backend["checkpoint"]].head
                output_dim = (
                    head[-1].out_features
                    if isinstance(head, nn.Sequential)
                    else head.out_features
                )
                if output_dim != len(self.classes[name]):
                    raise ValueError("原生分类头维度与配置的标签数量不一致")
            else:
                with backend["path"].open("rb") as stream:
                    estimator = compat.load(stream)
                if not isinstance(estimator, CLASSIFIER_TYPES[name]):
                    raise TypeError(f"{name} 的模型文件类型不匹配")
                estimator._check_is_fitted()
                self.estimators[name] = estimator
                self.classes[name] = np.asarray(estimator.classes_).tolist()
            if len(self.classes[name]) != 2 or set(self.classes[name]) != labels:
                raise ValueError(f"{name} 训练标签与配置不一致：{self.classes[name]}")
            LOGGER.info("已启用分类器：%s，异常阈值：%.6f", name, backend["threshold"])

    @torch.inference_mode()
    def score(self, waveform: np.ndarray) -> dict[str, float]:
        """对一个十秒窗口计算异常分数，原生头与 embedding 后端分别使用正确归一化。

        同一 checkpoint 只做一次 BEATs 前向；任一后端返回非法概率立即报错，
        不将推理失败误报为正常。返回字典保留 YAML 中的后端顺序。
        """
        pooled = {}
        features = None
        front = self.config["frontend"]
        for path, model in self.models.items():
            if features is None:
                features = self.frontend(waveform, model.backbone.input_patch_size)
            patch = model.backbone.input_patch_size
            tokens = (features.shape[1] // patch) * (features.shape[2] // patch)
            if min(features.shape[1:]) < patch or tokens > front["max_tokens"]:
                raise ValueError("前处理特征尺寸不满足当前 BEATs 模型的 token 限制")
            pooled[path] = model.embedding(features, front)
        embeddings = None
        if self.estimators:
            embeddings = pooled[self.embedding_path]
            if self.config["embedding"]["normalize"]:
                embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
            embeddings = embeddings.cpu().numpy()
        result = {}
        for name, backend in self.backends.items():
            if name == "beats-native":
                logits = self.models[backend["checkpoint"]].head(
                    pooled[backend["checkpoint"]]
                )
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                probabilities = np.asarray(
                    self.estimators[name].predict_proba(embeddings)
                )
            if (
                probabilities.shape != (1, 2)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            ):
                raise ValueError(f"分类器 {name} 返回非法概率")
            positive = self.classes[name].index(
                self.config["ensemble"]["abnormal_label"]
            )
            result[name] = float(probabilities[0, positive])
        return result
