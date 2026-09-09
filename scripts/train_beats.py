#!/usr/bin/env python3
"""根据 train_beats.yaml 执行 BEATs 分类头训练、全量微调或从零训练。"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from beats_trainer.core.config import Config, ModelConfig  # noqa: E402
from beats_trainer.training.trainer import BEATsTrainer  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).with_name("train_beats.yaml")
MODE_NAMES = {
    "head": "分类头训练",
    "finetune": "全量微调",
    "scratch": "从零训练",
}
MODE_FIELDS = {"train_from_scratch", "freeze_backbone", "fine_tune_backbone"}
SECTION_FIELDS = {
    "data": {"sample_rate", "batch_size", "num_workers"},
    "model": {item.name for item in fields(ModelConfig)}
    - MODE_FIELDS
    - {"num_classes"},
    "training": {
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "patience",
        "optimizer",
        "scheduler",
        "monitor_metric",
        "gpus",
        "precision",
        "log_every_n_steps",
        "save_top_k",
    },
}
DATA_PATH_FIELDS = {"train_dir", "val_dir", "test_dir"}


@dataclass(frozen=True)
class TrainingSettings:
    """保存已合并模式参数的训练配置及按 YAML 位置解析的绝对路径。"""

    mode: str
    config: Config
    train_dir: Path
    val_dir: Path
    test_dir: Path | None
    log_dir: Path
    test_after_training: bool
    resume_from_checkpoint: Path | None


def checked_mapping(value: Any, allowed: set[str], location: str) -> dict:
    """校验指定位置的 YAML 映射及允许字段，返回浅拷贝；类型或字段错误时抛出 ValueError。"""
    if not isinstance(value, dict):
        raise ValueError(f"{location} 必须为参数映射")
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"{location} 包含不支持的参数：{names}")
    return dict(value)


def resolve_path(value: Any, base_dir: Path, name: str) -> Path:
    """将非空路径按 YAML 所在目录解析为绝对路径；空值或非字符串参数抛出 ValueError。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须为非空路径字符串")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def validate_config(config: Config) -> None:
    """校验训练数值、枚举及随机初始化维度，提前拒绝无法执行的配置并给出中文原因。"""
    for section, bounds in (
        ("data", {"sample_rate": 1, "batch_size": 1, "num_workers": 0}),
        (
            "training",
            {
                "max_epochs": 1,
                "patience": 0,
                "gpus": 0,
                "log_every_n_steps": 1,
                "save_top_k": -1,
            },
        ),
    ):
        for name, minimum in bounds.items():
            value = getattr(getattr(config, section), name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{section}.{name} 必须为不小于 {minimum} 的整数")
    for name in ("learning_rate", "weight_decay"):
        value = getattr(config.training, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{name} 必须为有限的非负数")
    if config.training.learning_rate == 0:
        raise ValueError("training.learning_rate 必须大于 0")
    for name, options in (
        ("optimizer", ("adamw", "adam", "sgd")),
        ("scheduler", ("cosine", "step", "none")),
        ("monitor_metric", ("val_accuracy", "val_loss")),
        ("precision", (32, 16, "32-true", "16-mixed", "bf16-mixed")),
    ):
        if getattr(config.training, name) not in options:
            raise ValueError(f"training.{name} 可选值：{options}")
    if type(config.model.use_custom_head) is not bool:
        raise ValueError("model.use_custom_head 必须为 true 或 false")
    if config.model.activation not in ("relu", "gelu"):
        raise ValueError("model.activation 只能为 relu 或 gelu")
    dropout = config.model.dropout_rate
    if type(dropout) not in (int, float) or not 0 <= dropout < 1:
        raise ValueError("model.dropout_rate 必须在 [0, 1) 范围内")
    if config.model.use_custom_head:
        hidden_dims = config.model.hidden_dims
        if (
            not isinstance(hidden_dims, list)
            or not hidden_dims
            or any(type(size) is not int or size < 1 for size in hidden_dims)
        ):
            raise ValueError("启用自定义分类头时，model.hidden_dims 必须为正整数列表")
    if config.model.train_from_scratch:
        for name in (
            "encoder_layers",
            "encoder_embed_dim",
            "encoder_ffn_embed_dim",
            "encoder_attention_heads",
            "input_patch_size",
            "embed_dim",
        ):
            value = getattr(config.model, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"model.{name} 必须为正整数")
        if config.model.encoder_embed_dim % config.model.encoder_attention_heads:
            raise ValueError("encoder_embed_dim 必须能被 encoder_attention_heads 整除")
        if config.model.encoder_embed_dim % 16:
            raise ValueError("encoder_embed_dim 必须能被位置卷积的 16 个分组整除")


def load_config(
    config_path: Path, mode_override: str | None = None
) -> TrainingSettings:
    """读取 UTF-8 YAML，合并公共参数与所选模式参数，并生成可执行的训练配置。

    mode_override 非空时优先于 YAML 的 mode；模式自动控制冻结和随机初始化开关。
    所有相对路径以 YAML 所在目录为基准；未知字段和非法参数抛出 ValueError，
    文件读取及 YAML 解析错误向调用方传递。本函数不加载模型、不启动训练。
    """
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8-sig") as stream:
        values = checked_mapping(
            yaml.safe_load(stream),
            {
                "mode",
                "experiment_name",
                "log_dir",
                "test_after_training",
                "resume_from_checkpoint",
                "data",
                "model",
                "training",
                "modes",
            },
            "配置顶层",
        )
    mode = mode_override if mode_override is not None else values.get("mode", "head")
    if not isinstance(mode, str) or mode not in MODE_NAMES:
        raise ValueError("mode 只能为 head、finetune 或 scratch")
    profiles = checked_mapping(values.get("modes", {}), set(MODE_NAMES), "modes")
    for name, profile in profiles.items():
        profile = checked_mapping(profile, set(SECTION_FIELDS), f"modes.{name}")
        for section, parameters in profile.items():
            checked_mapping(
                parameters, SECTION_FIELDS[section], f"modes.{name}.{section}"
            )

    merged = {}
    for section, allowed in SECTION_FIELDS.items():
        section_allowed = allowed | DATA_PATH_FIELDS if section == "data" else allowed
        merged[section] = checked_mapping(
            values.get(section, {}), section_allowed, section
        )
        merged[section].update(profiles.get(mode, {}).get(section, {}))

    base_dir = config_path.parent
    paths = {}
    for name in sorted(DATA_PATH_FIELDS):
        value = merged["data"].pop(name, None)
        paths[name] = (
            None
            if name == "test_dir" and value is None
            else resolve_path(value, base_dir, f"data.{name}")
        )
    test_after_training = values.get("test_after_training", True)
    if type(test_after_training) is not bool:
        raise ValueError("test_after_training 必须为 true 或 false")
    if test_after_training and paths["test_dir"] is None:
        raise ValueError("启用训练后测试时，必须配置 data.test_dir")

    model = merged["model"]
    model.update(
        train_from_scratch=mode == "scratch",
        freeze_backbone=mode == "head",
        fine_tune_backbone=mode != "head",
    )
    if mode == "scratch":
        model["model_path"] = None
    else:
        model_path = model.get("model_path")
        model["model_path"] = str(
            resolve_path(
                model_path
                if model_path is not None
                else str(PROJECT_ROOT / "checkpoints" / "BEATs_iter3_plus_AS2M.pt"),
                base_dir,
                "model.model_path",
            )
        )
    experiment_name = values.get("experiment_name", "mimii_pump")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment_name 必须为非空字符串")
    config = Config.from_dict(
        {**merged, "experiment_name": f"{experiment_name}_{mode}"}
    )
    validate_config(config)
    resume = values.get("resume_from_checkpoint")
    return TrainingSettings(
        mode=mode,
        config=config,
        **paths,
        log_dir=resolve_path(values.get("log_dir", "../logs"), base_dir, "log_dir"),
        test_after_training=test_after_training,
        resume_from_checkpoint=(
            resolve_path(resume, base_dir, "resume_from_checkpoint")
            if resume is not None
            else None
        ),
    )


def run_training(settings: TrainingSettings) -> None:
    """校验数据和恢复点路径，创建训练器并训练；按配置选择是否使用最佳可用检查点测试。

    settings 必须由 load_config 生成；恢复路径会传给 Lightning 以恢复完整训练状态。
    缺失路径抛出 FileNotFoundError，模型加载和训练异常由调用方处理。
    """
    for name in ("train_dir", "val_dir", "test_dir"):
        path = getattr(settings, name)
        if path is not None and not path.is_dir():
            raise FileNotFoundError(f"{name} 数据目录不存在：{path}")
    resume = settings.resume_from_checkpoint
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(f"恢复训练的检查点不存在：{resume}")
    LOGGER.info(
        "开始%s，实验名称：%s",
        MODE_NAMES[settings.mode],
        settings.config.experiment_name,
    )
    trainer = BEATsTrainer.from_split_directories(
        train_dir=str(settings.train_dir),
        val_dir=str(settings.val_dir),
        test_dir=str(settings.test_dir) if settings.test_dir is not None else None,
        config=settings.config,
        log_dir=str(settings.log_dir),
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
    if settings.test_after_training:
        LOGGER.info("训练完成，开始使用最佳可用检查点评估测试集")
        trainer.test()
    LOGGER.info(
        "%s完成，日志与检查点根目录：%s", MODE_NAMES[settings.mode], settings.log_dir
    )


def main() -> int:
    """解析命令行并加载配置，执行训练或打印有效配置；成功返回 0，配置或路径错误返回 1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="YAML 配置路径"
    )
    parser.add_argument(
        "--mode", choices=tuple(MODE_NAMES), help="临时覆盖 YAML 中的训练模式"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="仅校验并打印最终配置，不加载模型或训练"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s：%(message)s")
    try:
        settings = load_config(args.config, args.mode)
        if args.dry_run:
            effective = asdict(settings)
            for name in (*DATA_PATH_FIELDS, "log_dir", "resume_from_checkpoint"):
                effective[name] = (
                    str(effective[name]) if effective[name] is not None else None
                )
            print(yaml.safe_dump(effective, allow_unicode=True, sort_keys=False))
        else:
            run_training(settings)
    except (OSError, ValueError, yaml.YAMLError) as error:
        LOGGER.error("训练未完成：%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
