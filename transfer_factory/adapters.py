"""BEATs、TorchScript 及自定义 PyTorch 模型的导出适配器。"""

from importlib import import_module

import torch
from torch import nn


class BEATsExportModel(nn.Module):
    """导出 BEATs 特征模型与可选原生分类头，输入为尚未标准化的时频特征。"""

    def __init__(self, config: dict):
        """复用 ASD 离线加载器重建网络，校验固定输入尺寸及分类头标签数量。"""
        super().__init__()
        from asd.models import BEATsModel

        model = config["model"]
        loaded = BEATsModel(
            model["checkpoint"],
            torch.device("cpu"),
            model["backbone_overrides"],
            model["output_mode"] != "embedding",
        )
        self.backbone = loaded.backbone
        self.head = loaded.head
        self.mode = model["output_mode"]
        self.normalize = model["normalize_embedding"]
        self.mean = model["fbank_mean"]
        self.std = model["fbank_std"]
        _, time, frequency = config["input"]["shape"]
        patch = self.backbone.input_patch_size
        tokens = (time // patch) * (frequency // patch)
        if min(time, frequency) < patch or tokens > model["max_tokens"]:
            raise ValueError(f"输入特征尺寸不能用于 BEATs 导出，token 数：{tokens}")
        if self.head is not None:
            count = (
                self.head[-1].out_features
                if isinstance(self.head, nn.Sequential)
                else self.head.out_features
            )
            if len(model["labels"]) != count or len(set(model["labels"])) != count:
                raise ValueError("labels 数量或顺序声明不满足原生分类头要求")

    def forward(self, features: torch.Tensor):
        """返回平均 embedding 和/或原生头 softmax；分类头始终使用未 L2 归一化特征。"""
        sequence, _ = self.backbone.extract_features(
            features, fbank_mean=self.mean, fbank_std=self.std
        )
        pooled = sequence.mean(dim=1)
        embedding = (
            nn.functional.normalize(pooled, p=2, dim=-1) if self.normalize else pooled
        )
        if self.mode == "embedding":
            return embedding
        probabilities = torch.softmax(self.head(pooled), dim=-1)
        return (
            probabilities
            if self.mode == "probabilities"
            else (embedding, probabilities)
        )


def load_model(config: dict) -> nn.Module:
    """按 kind 加载 CPU/eval 模型；自定义网络必须显式提供构造函数并严格加载 state_dict。

    .pt 扩展名不能唯一确定结构，因此不自动猜测任意 pickle 内的网络。
    BEATs 和 python_factory checkpoint 仅可来自可信来源。
    """
    model = config["model"]
    if not model["checkpoint"].is_file():
        raise FileNotFoundError(f"模型文件不存在：{model['checkpoint']}")
    if model["kind"] == "beats":
        network = BEATsExportModel(config)
    elif model["kind"] == "torchscript":
        network = torch.jit.load(str(model["checkpoint"]), map_location="cpu")
    else:
        if not isinstance(model["factory"], str) or ":" not in model["factory"]:
            raise ValueError("python_factory 必须配置 module:function")
        module, name = model["factory"].rsplit(":", 1)
        network = getattr(import_module(module), name)(**model["factory_kwargs"])
        if not isinstance(network, nn.Module):
            raise TypeError("自定义构造函数必须返回 torch.nn.Module")
        state = torch.load(model["checkpoint"], map_location="cpu", weights_only=False)
        if model["state_dict_key"] is not None:
            state = state[model["state_dict_key"]]
        network.load_state_dict(state, strict=True)
    return network.cpu().eval()
