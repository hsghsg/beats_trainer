"""验证 YAML 训练入口的模式行为、路径解析、执行流程及 CPU/GPU 参数。"""

import copy
from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from scripts import train_beats
from beats_trainer.core.model import BEATsLightningModule
from beats_trainer.training.callbacks import setup_pytorch_lightning_trainer

SMALL_MODEL = {
    "encoder_layers": 1,
    "encoder_embed_dim": 32,
    "encoder_ffn_embed_dim": 64,
    "encoder_attention_heads": 4,
    "embed_dim": 32,
    "input_patch_size": 16,
}


@pytest.fixture
def config_file(tmp_path):
    """将正式 YAML 复制到临时目录并替换资源路径，返回配置文件和可修改参数。"""
    values = yaml.safe_load(train_beats.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    values["log_dir"] = "logs"
    values["data"].update(
        train_dir="train", val_dir="val", test_dir="test", num_workers=0
    )
    values["model"]["model_path"] = "tiny.pt"
    values["training"]["gpus"] = 0
    path = tmp_path / "train_beats.yaml"
    write_config(path, values)
    return path, values


def write_config(path, values):
    """以 UTF-8 写入用例 YAML，供真实配置加载器解析；仅操作测试临时文件。"""
    path.write_text(yaml.safe_dump(values, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """创建一层 BEATs 的本地权重用于真实加载测试，避免下载及占用大型模型资源。"""
    config = train_beats.Config.from_dict(
        {
            "model": {**SMALL_MODEL, "train_from_scratch": True},
        }
    )
    model = BEATsLightningModule(config, num_classes=2)
    checkpoint = {
        "cfg": vars(model.backbone.cfg),
        "model": model.backbone.state_dict(),
    }
    path = tmp_path / "tiny.pt"
    torch.save(checkpoint, path)
    return checkpoint


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    """限制微型模型测试的 CPU 线程数，并在用例结束后恢复调用环境。"""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("mode", "batch", "lr", "epochs"),
    [("head", 16, 1e-4, 30), ("finetune", 4, 5e-5, 50), ("scratch", 4, 1e-3, 100)],
)
def test_profile_and_paths_are_independent_of_cwd(
    config_file, tmp_path, monkeypatch, mode, batch, lr, epochs
):
    """从其他工作目录加载正式模式配置，验证覆盖结果、绝对路径和实验名区分。"""
    path, values = config_file
    values["mode"] = mode
    write_config(path, values)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    settings = train_beats.load_config(path)
    assert settings.mode == mode
    assert settings.train_dir == tmp_path / "train"
    assert settings.log_dir == tmp_path / "logs"
    assert settings.config.data.batch_size == batch
    assert settings.config.training.learning_rate == lr
    assert settings.config.training.max_epochs == epochs
    assert settings.config.experiment_name == f"mimii_pump_{mode}"
    assert settings.config.model.model_path == (
        None if mode == "scratch" else str(tmp_path / "tiny.pt")
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("root", "mode", "unknown", "mode"),
        ("root", "test_after_training", "false", "test_after_training"),
        ("model", "freeze_backbone", False, "freeze_backbone"),
        ("data", "batch_size", 0, "batch_size"),
        ("training", "gpus", -1, "gpus"),
        ("training", "max_epoch", 3, "max_epoch"),
        ("training", "scheduler", "plateau", "scheduler"),
        ("model", "dropout_rate", 1, "dropout_rate"),
    ],
)
def test_invalid_configuration_fails_early(config_file, section, key, value, message):
    """验证错拼参数、模式冲突、非法数值和字符串布尔值在模型加载前被拒绝。"""
    path, values = config_file
    target = values if section == "root" else values[section]
    target[key] = value
    write_config(path, values)
    with pytest.raises(ValueError, match=message):
        train_beats.load_config(path)


def test_dry_run_overrides_mode_without_training(config_file, monkeypatch, capsys):
    """验证命令行覆盖 YAML 模式，并确保 dry-run 不构造模型、不下载权重或写日志目录。"""
    path, _ = config_file
    factory = Mock()
    monkeypatch.setattr(train_beats.BEATsTrainer, "from_split_directories", factory)
    monkeypatch.setattr(
        "sys.argv",
        ["train_beats.py", "--config", str(path), "--mode", "scratch", "--dry-run"],
    )
    assert train_beats.main() == 0
    effective = yaml.safe_load(capsys.readouterr().out)
    assert effective["mode"] == "scratch"
    assert effective["config"]["model"]["train_from_scratch"] is True
    assert effective["config"]["model"]["model_path"] is None
    factory.assert_not_called()
    assert not (path.parent / "logs").exists()


def test_resume_and_optional_test_are_forwarded(config_file, monkeypatch):
    """验证恢复检查点传入训练器，并且关闭训练后测试时允许省略测试目录。"""
    path, values = config_file
    for name in ("train", "val"):
        (path.parent / name).mkdir()
    resume = path.parent / "last.ckpt"
    resume.touch()
    values.update(test_after_training=False, resume_from_checkpoint="last.ckpt")
    values["data"]["test_dir"] = None
    write_config(path, values)
    factory = Mock()
    monkeypatch.setattr(train_beats.BEATsTrainer, "from_split_directories", factory)
    train_beats.run_training(train_beats.load_config(path))
    assert factory.call_args.kwargs["test_dir"] is None
    factory.return_value.train.assert_called_once_with(
        resume_from_checkpoint=str(resume)
    )
    factory.return_value.test.assert_not_called()


def test_missing_data_stops_before_model_loading(config_file, monkeypatch):
    """验证数据路径不存在时立即报错，避免开始模型下载或训练器初始化。"""
    path, _ = config_file
    factory = Mock()
    monkeypatch.setattr(train_beats.BEATsTrainer, "from_split_directories", factory)
    with pytest.raises(FileNotFoundError, match="train_dir"):
        train_beats.run_training(train_beats.load_config(path))
    factory.assert_not_called()


@pytest.mark.parametrize("mode", ["head", "finetune", "scratch"])
def test_real_optimizer_updates_only_expected_parameters(
    config_file, tiny_checkpoint, mode
):
    """用真实 BEATs 前向、反向及优化步骤验证冻结策略，且从零模式禁止读取任何权重文件。"""
    path, values = config_file
    values["modes"]["scratch"]["model"] = copy.deepcopy(SMALL_MODEL)
    values["training"]["scheduler"] = "none"
    write_config(path, values)
    config = train_beats.load_config(path, mode).config
    if mode == "scratch":
        with patch("torch.load", side_effect=AssertionError("从零训练不应加载权重")):
            model = BEATsLightningModule(config, num_classes=2)
    else:
        model = BEATsLightningModule(config, num_classes=2)
        assert torch.equal(
            model.backbone.patch_embedding.weight,
            tiny_checkpoint["model"]["patch_embedding.weight"],
        )
    backbone_before = model.backbone.patch_embedding.weight.detach().clone()
    classifier_before = model.classifier.weight.detach().clone()
    optimizer = model.configure_optimizers()
    optimizer.zero_grad()
    logits = model(torch.randn(2, 8000))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    optimizer.step()
    assert not torch.equal(classifier_before, model.classifier.weight)
    assert torch.equal(backbone_before, model.backbone.patch_embedding.weight) == (
        mode == "head"
    )
    if mode == "head":
        assert all(parameter.grad is None for parameter in model.backbone.parameters())
    else:
        assert model.backbone.patch_embedding.weight.grad is not None


@pytest.mark.parametrize(
    ("gpus", "accelerator", "devices"), [(0, "cpu", 1), (2, "gpu", 2)]
)
def test_device_selection_respects_configuration(
    config_file, monkeypatch, gpus, accelerator, devices
):
    """模拟 CUDA 可用的机器，回归验证强制 CPU 以及指定 GPU 数量都不会被自动选择覆盖。"""
    path, _ = config_file
    config = train_beats.load_config(path).config
    config.training.gpus = gpus
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=True))
    trainer = Mock()
    monkeypatch.setattr("pytorch_lightning.Trainer", trainer)
    monkeypatch.setattr("pytorch_lightning.loggers.TensorBoardLogger", Mock())
    setup_pytorch_lightning_trainer(config, [], path.parent / "logs")
    assert trainer.call_args.kwargs["accelerator"] == accelerator
    assert trainer.call_args.kwargs["devices"] == devices


def test_unavailable_gpu_is_reported(config_file, monkeypatch):
    """验证请求 GPU 但 CUDA 不可用时给出可操作的配置错误，而非悄悄改用 CPU。"""
    path, _ = config_file
    config = train_beats.load_config(path).config
    config.training.gpus = 1
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    with pytest.raises(ValueError, match="training.gpus"):
        setup_pytorch_lightning_trainer(config, [], path.parent / "logs")


def test_scratch_training_and_test_complete_on_tiny_audio(config_file):
    """以临时双类别音频执行一轮真实 CPU 训练、检查点保存和测试集评估，验证完整调用链。"""
    path, values = config_file
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        for label in ("normal", "abnormal"):
            directory = path.parent / split / label
            directory.mkdir(parents=True)
            sf.write(directory / "sample.wav", rng.normal(0, 0.1, 8000), 16000)
    values["modes"]["scratch"] = {
        "model": copy.deepcopy(SMALL_MODEL),
        "data": {"batch_size": 2},
        "training": {"learning_rate": 1e-3, "max_epochs": 1, "patience": 1},
    }
    values["training"]["log_every_n_steps"] = 1
    write_config(path, values)
    settings = train_beats.load_config(path, "scratch")
    train_beats.run_training(settings)
    checkpoints = list(settings.log_dir.rglob("*.ckpt"))
    assert any(checkpoint.name == "last.ckpt" for checkpoint in checkpoints)
    assert any("mimii_pump_scratch" in checkpoint.name for checkpoint in checkpoints)
