"""复用训练项目的 log-mel 和连续小波前处理。"""

import torch
from torchaudio.compliance import kaldi

from .vendor.beats_trainer.data.preprocessing import CWTAmplitudeTransform


class Frontend:
    """将十秒 16 kHz 波形转换为尚未标准化的 [1,time,freq] 特征图。"""

    def __init__(self, config: dict):
        """保存前处理配置；小波算法采用项目现有 CWTAmplitudeTransform 副本。"""
        self.config = config
        self.cwt = CWTAmplitudeTransform(16000, config["cwt_voices_per_octave"])

    def __call__(self, waveform, patch_size: int) -> torch.Tensor:
        """计算特征并检查 patch/token 长度；归一化由 BEATs 执行一次。

        log-mel 使用项目相同的 Kaldi 参数；wavelet 支持显式时间抽样。
        超过 max_tokens 时抛出 ValueError，不擅自截断或改变训练分布。
        """
        source = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
        if (
            source.ndim != 1
            or source.numel() != 160000
            or not torch.isfinite(source).all()
        ):
            raise ValueError("前处理输入必须为十秒 16 kHz 有限单声道波形")
        if self.config["method"] == "log-mel-energy":
            features = kaldi.fbank(
                source.unsqueeze(0) * 2**15,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
        else:
            features = self.cwt(source)[:: self.config["cwt_time_stride"]]
        tokens = (features.shape[0] // patch_size) * (features.shape[1] // patch_size)
        if min(features.shape) < patch_size or tokens > self.config["max_tokens"]:
            raise ValueError(
                f"前处理特征形状 {tuple(features.shape)} 产生 {tokens} 个 token，"
                f"上限为 {self.config['max_tokens']}。请使用与训练一致的前处理/抽样参数，"
                "或在确认内存容量后调整 max_tokens"
            )
        if not torch.isfinite(features).all():
            raise ValueError("前处理产生了非有限值")
        return features.unsqueeze(0)
