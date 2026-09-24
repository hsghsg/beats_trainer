"""回归验证小波时频输入的数值、时间聚合、分类训练及文件扫描行为。"""

import math

import pytest
import torch

from beats_trainer.core.config import Config, DataConfig
from beats_trainer.core.model import BEATsLightningModule
from beats_trainer.data.datasets import scan_directory_dataset
from beats_trainer.data.module import collate_audio_batch
from beats_trainer.data.preprocessing import CWTAmplitudeTransform, create_audio_preprocessor
from scripts.train_beats import validate_config


def test_cwt_preserves_tone_frequency_and_partial_frame():
    """用已知正弦波验证 CWT 峰值频率，并验证聚合保留不足一帧的尾部幅值。"""
    sample_rate = 16000
    waveform = torch.sin(2 * math.pi * 500 * torch.arange(16017) / sample_rate)
    raw = CWTAmplitudeTransform(sample_rate)(waveform)
    framed = CWTAmplitudeTransform(sample_rate, frame_hop=160)(waveform)
    assert framed.shape == (101, 91)
    torch.testing.assert_close(framed[-1], raw[-17:].mean(dim=0))
    peak = raw[4000:12000].mean(dim=0).argmax().item()
    frequency = 6 * sample_rate / (2 * math.pi * 2.78896039134963 * 2 ** (peak / 12))
    assert abs(frequency - 500) < 40


def test_normalized_cwt_is_finite_for_signal_and_silence():
    """验证音频标准化后的均值方差及静音数值稳定性，防止训练出现 NaN。"""
    transform = create_audio_preprocessor(DataConfig(
        audio_preprocess="cwt", cwt_frame_hop=160, cwt_log_normalize=True,
    ))
    generator = torch.Generator().manual_seed(42)
    result = transform(torch.randn(16000, generator=generator))
    assert result.shape == (100, 91)
    assert torch.isfinite(result).all()
    assert abs(result.mean().item()) < 1e-5
    assert abs(result.std(unbiased=False).item() - 1) < 1e-5
    silence = transform(torch.zeros(16000))
    assert torch.isfinite(silence).all()
    assert silence.abs().max() < 1e-5


def test_cwt_batch_updates_real_beats_weights():
    """将不同长度 CWT 特征组成批次，验证真实 BEATs 前后向及主干权重更新。"""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = Config.from_dict({
            "data": {"audio_preprocess": "cwt", "cwt_frame_hop": 160,
                     "cwt_log_normalize": True},
            "model": {"train_from_scratch": True, "encoder_layers": 1,
                      "encoder_embed_dim": 32, "encoder_ffn_embed_dim": 64,
                      "encoder_attention_heads": 4, "embed_dim": 32},
            "training": {"scheduler": "none", "gpus": 0},
        })
        transform = create_audio_preprocessor(config.data)
        features = [transform(torch.randn(length)) for length in (8000, 10240)]
        x, mask, labels = collate_audio_batch([
            (item, torch.zeros(item.shape[0], dtype=torch.bool), label)
            for label, item in enumerate(features)
        ])
        assert mask[0, 50:].all()
        model = BEATsLightningModule(config, num_classes=2)
        before = model.backbone.patch_embedding.weight.detach().clone()
        optimizer = model.configure_optimizers()
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        assert not torch.equal(before, model.backbone.patch_embedding.weight)
        torch.testing.assert_close(
            model.backbone.preprocess(x, fbank_mean=0.0, fbank_std=0.5), x,
        )
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("field,value", [
    ("cwt_frame_hop", 0), ("cwt_frame_hop", True),
    ("cwt_log_normalize", "true"),
])
def test_invalid_cwt_options_fail_before_training(field, value):
    """确保非法小波参数在加载模型和占用 GPU 之前被拒绝。"""
    config = Config()
    setattr(config.data, field, value)
    with pytest.raises(ValueError, match=field):
        validate_config(config)


def test_directory_scan_includes_each_file_once(tmp_path):
    """跨平台验证大小写扩展名均被识别，且文件不会因重复扩展名被计入多次。"""
    directory = tmp_path / "normal"
    directory.mkdir()
    for name in ("first.wav", "second.WAV", "third.WaV", "ignored.txt"):
        (directory / name).touch()
    frame = scan_directory_dataset(tmp_path, [".wav", ".WAV"])
    assert len(frame) == 3
    assert frame.filename.nunique() == 3
