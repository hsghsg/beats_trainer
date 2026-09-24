"""音频前处理工具（可选支持连续小波变换幅值时频图）。"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..core.config import DataConfig

LOGGER = logging.getLogger(__name__)


def _build_wavelet_scale_params(voices_per_octave: int) -> tuple[float, float]:
    """根据倍频程返回 CWT 尺度构造参数，保持与外部 C# 实现一致。"""
    if voices_per_octave == 4:
        return 2.95480023611212, 7.51586947392636 * voices_per_octave
    if voices_per_octave == 24:
        return 2.70956077884719, 7.50000099529183 * voices_per_octave
    return 2.78896039134963, 7.58333225642125 * voices_per_octave


class CWTAmplitudeTransform:
    """连续小波变换（基于外部项目 CWTFT）幅值时频图前处理器。"""

    def __init__(
        self,
        sample_rate: int,
        voices_per_octave: int = 12,
        save_visualization: bool = False,
        visualize_dir: Optional[str | Path] = None,
        max_visualize_count: int = 4,
        frame_hop: int = 1,
        log_normalize: bool = False,
    ):
        """初始化 CWT 前处理器。

        Args:
            sample_rate: 输入音频采样率。
            voices_per_octave: 每倍频程的小波尺度个数。
            save_visualization: 是否在前若干次调用中输出时频图图片。
            visualize_dir: 时频图图片输出目录。
            max_visualize_count: 最多保存多少张图片。
            frame_hop: 每帧平均的采样点数，末尾不足一帧时仅平均有效点。
            log_normalize: 是否对帧幅值取自然对数并按单条音频标准化。

        Raises:
            ValueError: frame_hop 不是正整数或 log_normalize 不是布尔值。
        """
        self.sample_rate = float(sample_rate)
        self.voices_per_octave = int(voices_per_octave)
        self.save_visualization = bool(save_visualization)
        self.visualize_dir = Path(visualize_dir) if visualize_dir else None
        if type(max_visualize_count) is not int or max_visualize_count < 0:
            max_visualize_count = 0
        self.max_visualize_count = max_visualize_count
        self._visualized_count = 0
        if type(frame_hop) is not int or frame_hop < 1:
            raise ValueError("CWT 帧步长必须为正整数")
        if type(log_normalize) is not bool:
            raise ValueError("CWT 对数标准化开关必须为布尔值")
        self.frame_hop = frame_hop
        self.log_normalize = log_normalize

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """将 1D 波形张量转换为 2D 幅值时频图。

        返回张量形状为 [time, freq]，与现有 BEATs fbank 输入形状一致，
        后续可直接用于 `padding_mask` 与 `collate`。
        """
        if not torch.is_tensor(waveform):
            raise TypeError("CWT 前处理器要求输入为 torch.Tensor")
        if waveform.ndim != 1:
            raise ValueError("CWT 前处理器只支持 1D 波形输入")

        signal = waveform.detach().to(dtype=torch.float64, device="cpu").numpy()
        result = self._compute_cwtamplitude(signal)
        if self.log_normalize and result.size:
            result = np.log(np.maximum(result, 1e-8))
            result = (result - result.mean()) / max(float(result.std()), 1e-6)
        cwt = torch.from_numpy(result).to(dtype=torch.float32, device=waveform.device)
        if self.save_visualization:
            self._save_visualization_if_needed(cwt)
        return cwt

    def _save_visualization_if_needed(self, cwt: torch.Tensor) -> None:
        """在启用自动绘图时保存前若干份 CWT 矩阵；缺少绘图库时记录日志并停止绘图。"""
        if self.visualize_dir is None or cwt.numel() == 0:
            return
        if self._visualized_count >= self.max_visualize_count:
            return
        output_file = self.visualize_dir / f"cwt_amp_{self._visualized_count + 1:03d}.png"
        try:
            self.visualize(cwt, output_file)
        except ImportError:
            LOGGER.warning("未安装 matplotlib，无法输出 CWT 可视化图像")
            self.max_visualize_count = 0
            return
        self._visualized_count += 1

    def visualize(self, cwt: torch.Tensor, output_path: str | Path) -> Path:
        """将本实例生成的 CWT 幅值矩阵保存为 PNG 时频图。

        Args:
            cwt: [time, scale] 幅值张量，至少包含两个时间点。
            output_path: PNG 保存路径；不存在的父目录会自动创建。

        Returns:
            已保存图片的绝对路径。横轴为秒，纵轴为 Morlet 中心频率
            6 / (2 * pi * scale) 对应的近似频率；颜色由 log_normalize 决定。

        Raises:
            ValueError: 矩阵维度、尺度数量或输出扩展名不符合要求。
            ImportError: 未安装 matplotlib。
            OSError: 无法创建目录或写入图片。
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        s0, len_scale = _build_wavelet_scale_params(self.voices_per_octave)
        scale_count = int(math.floor(len_scale)) + 1
        if cwt.ndim != 2 or cwt.shape[0] < 2 or cwt.shape[1] != scale_count:
            raise ValueError("CWT 图像要求至少两个时间点，且尺度数量须与前处理器一致")
        output_file = Path(output_path).expanduser().resolve()
        if output_file.suffix.lower() != ".png":
            raise ValueError("CWT 图像输出路径必须使用 .png 扩展名")
        scales = (s0 / self.sample_rate) * 2.0 ** (
            np.arange(scale_count) / self.voices_per_octave
        )
        frequencies = 6.0 / (2.0 * np.pi * scales)
        times = np.arange(cwt.shape[0]) * self.frame_hop / self.sample_rate
        data = cwt.detach().cpu().numpy().T
        figure = Figure(figsize=(10, 4), layout="constrained")
        FigureCanvasAgg(figure)
        try:
            axes = figure.subplots()
            plot = axes.pcolormesh(
                times, frequencies[::-1], data[::-1], shading="nearest", cmap="jet"
            )
            axes.set_yscale("log")
            axes.set_xlabel("Time (s)")
            axes.set_ylabel("Approx. frequency (Hz)")
            axes.set_title("CWT spectrogram")
            color_label = "Normalized log amplitude" if self.log_normalize else "Amplitude"
            figure.colorbar(plot, ax=axes, label=color_label)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_file, dpi=160)
        finally:
            figure.clear()
        LOGGER.info("CWT 幅值时频图已保存：%s", output_file)
        return output_file


    def _compute_cwtamplitude(self, signal: np.ndarray) -> np.ndarray:
        """执行连续小波变换并返回幅值矩阵。

        该实现逐步对应外部 C# 代码 `CWTFT.Execute` 的流程：
        1) 去均值、
        2) 补零到 2 的幂、
        3) 构建 omega 频率轴、
        4) 每个尺度执行 FFT 频域卷积与 IFFT，
        5) 保留原始信号长度的复数系数幅值，按 frame_hop 做不重叠帧平均。
        返回 [ceil(采样点数 / frame_hop), 尺度数]；空信号返回空矩阵。
        """
        signal = np.asarray(signal, dtype=np.float64)
        sample_count = int(signal.size)
        if sample_count <= 0:
            return np.zeros((0, 0), dtype=np.float64)

        # 去均值
        signal = signal - float(signal.mean())

        # 与 C# 一致的尺度参数
        s0, len_scale = _build_wavelet_scale_params(self.voices_per_octave)
        s0 /= self.sample_rate
        a0 = 2 ** (1.0 / self.voices_per_octave)
        scales = [s0 * (a0 ** i) for i in range(int(math.floor(len_scale)) + 1)]

        # 补零到 2 的幂长度
        log2_value = math.log(sample_count, 2) + 0.4999
        np2 = 1 + (math.floor(log2_value) if log2_value > 0 else math.ceil(log2_value))
        pad_count = int(math.pow(2, np2) - sample_count)
        x = np.concatenate([signal, np.zeros(pad_count, dtype=np.float64)], axis=0)

        n = x.size
        dt = 1.0 / self.sample_rate
        fixn = math.floor(n / 2.0)
        md = (2.0 * np.pi) / (n * dt)

        # 生成 omega 采样序列
        omega_values = [0.0]
        for i in range(1, int(fixn) + 1):
            omega_values.append(i * md)
        fixnm = int(math.floor((n - 1) / 2.0) - 1)
        for i in range(fixnm, -1, -1):
            omega_values.append(-omega_values[i])
        omega = np.asarray(omega_values, dtype=np.float64)
        if omega.size < 2:
            return np.zeros((0, 0), dtype=np.float64)

        spectrum = np.fft.fft(x)

        scale_num = len(scales)
        output_time = (sample_count + self.frame_hop - 1) // self.frame_hop
        result = np.zeros((scale_num, output_time), dtype=np.float64)
        frame_starts = np.arange(0, sample_count, self.frame_hop)
        frame_lengths = np.minimum(self.frame_hop, sample_count - frame_starts)

        stp_frq = omega[1]
        cfs_norm = math.sqrt(stp_frq) * math.sqrt(omega.size)
        mul = 0.751125544464943 * cfs_norm
        scale_normalizer = 25.7565252381060

        for row, scale in enumerate(scales):
            p = (scale * omega) - 6.0
            t = mul * np.sqrt(scale) * np.exp(-(p * p) / 2.0)
            coeff = np.fft.ifft(spectrum * t)
            amplitude = np.abs(coeff[:sample_count]) / scale_normalizer
            if self.frame_hop == 1:
                result[row] = amplitude
            else:
                result[row] = np.add.reduceat(amplitude, frame_starts) / frame_lengths

        # C# 的 result 是 [scale, time]，转置为 [time, scale] 与 BEATs 习惯一致
        return result.T


def create_audio_preprocessor(
    config: DataConfig,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """按 DataConfig 组装音频前处理函数。

    参数 `audio_preprocess` 支持：
    - waveform：不做额外处理。
    - cwt：连续小波变换幅值时频图。

    返回 `None` 表示不注入 transform；否则返回一个可调用对象。
    """
    method = (config.audio_preprocess or "waveform").strip().lower()

    if method in ("waveform", "raw", ""):
        return None

    if method == "cwt":
        return CWTAmplitudeTransform(
            sample_rate=config.sample_rate,
            voices_per_octave=config.cwt_voices_per_octave,
            frame_hop=config.cwt_frame_hop,
            log_normalize=config.cwt_log_normalize,
        )

    raise ValueError(f"不支持的音频前处理方式：{method}")
