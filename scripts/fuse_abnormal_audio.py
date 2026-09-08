#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""融合泵和电机异常录音，输出 PCM 音频及可追溯参数清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly

METHODS = ("weighted", "snr", "crossfade", "intermittent")
SILENCE_THRESHOLD = 1e-10


def rms(audio: np.ndarray) -> float:
    """计算非空一维波形的均方根幅度，返回线性幅值而非分贝。"""
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def file_digest(path: Path) -> str:
    """分块计算文件 SHA-256，供输入来源和生成结果追溯使用。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_audio(
    path: Path,
    sample_rate: int,
    start: float,
    duration: float,
    channel: int,
) -> tuple[np.ndarray, dict]:
    """按秒截取指定通道，去直流并抗混叠重采样，返回波形和来源元数据。

    channel 从零计数，-1 表示通道平均；无效参数、过短、静音、非有限
    波形和超出原始录音的截取请求均抛出 ValueError，不循环填充录音。
    """
    if (
        sample_rate <= 0
        or not math.isfinite(start)
        or start < 0
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("采样率和时长须为正，起点须非负且有限")
    with sf.SoundFile(path) as stream:
        source_rate, channels, frames = (
            stream.samplerate,
            stream.channels,
            stream.frames,
        )
        start_frame, count = round(start * source_rate), round(duration * source_rate)
        if start_frame >= frames or count < 2 or start_frame + count > frames:
            raise ValueError(f"截取范围无效或剩余音频不足：{path}")
        if channel < -1 or channel >= channels:
            raise ValueError(f"通道号无效：{channel}，文件通道数：{channels}")
        stream.seek(start_frame)
        samples = stream.read(count, dtype="float64", always_2d=True)
    audio = samples.mean(axis=1) if channel == -1 else samples[:, channel]
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"音频包含无效数值：{path}")
    dc_offset = float(audio.mean())
    audio = audio - dc_offset
    divisor = math.gcd(source_rate, sample_rate)
    if source_rate != sample_rate:
        audio = resample_poly(audio, sample_rate // divisor, source_rate // divisor)
    if rms(audio) <= SILENCE_THRESHOLD:
        raise ValueError(f"音频有效信号为静音：{path}")
    return audio, {
        "path": str(path.resolve()),
        "sha256": file_digest(path),
        "source_sample_rate": source_rate,
        "source_channels": channels,
        "source_frames": frames,
        "channel": channel,
        "start_frame": start_frame,
        "read_frames": count,
        "dc_removed": dc_offset,
    }


def mix_components(
    pump: np.ndarray,
    motor: np.ndarray,
    method: str,
    sample_rate: int,
    alpha: float = 0.5,
    snr_db: float = 0.0,
    period: float = 2.0,
    duty_cycle: float = 0.5,
    fade_seconds: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """对等长一维非静音输入计算泵、电机两路贡献及方法参数。

    weighted 使用 alpha 泵权重；snr 控制泵/电机功率比；crossfade 从泵
    等功率渐变至电机；intermittent 在持续泵声上周期性叠加带余弦淡变的
    电机。SNR 正值表示泵更强，间歇模式按开启区间计算 SNR。
    """
    if (
        pump.ndim != 1
        or motor.ndim != 1
        or pump.shape != motor.shape
        or pump.size < 2
        or sample_rate <= 0
        or not np.all(np.isfinite(pump))
        or not np.all(np.isfinite(motor))
    ):
        raise ValueError("输入必须为有限值、等长且至少两个采样点的一维音频")
    if min(rms(pump), rms(motor)) <= SILENCE_THRESHOLD:
        raise ValueError("无法融合静音信号")
    if method == "weighted":
        if not 0 < alpha < 1:
            raise ValueError("alpha 必须在 (0, 1) 内，保证两路均参与融合")
        return alpha * pump, (1 - alpha) * motor, {"alpha": alpha}
    if method == "crossfade":
        phase = np.linspace(0, np.pi / 2, pump.size)
        return (
            pump * np.cos(phase),
            motor * np.sin(phase),
            {
                "curve": "equal_power",
                "direction": "pump_to_motor",
            },
        )
    if method not in ("snr", "intermittent"):
        raise ValueError(f"不支持的融合方法：{method}")
    if not math.isfinite(snr_db) or abs(snr_db) > 80:
        raise ValueError("SNR 必须为 [-80, 80] 范围内的有限分贝值")
    mask = np.ones(pump.size, dtype=np.float64)
    parameters = {"target_pump_to_motor_db": snr_db}
    if method == "intermittent":
        if (
            not math.isfinite(period)
            or period <= 0
            or not 0 < duty_cycle < 1
            or not math.isfinite(fade_seconds)
            or fade_seconds <= 0
        ):
            raise ValueError("周期、淡变时长须为正，占空比须在 (0, 1) 内")
        period_samples = round(period * sample_rate)
        active = round(period_samples * duty_cycle)
        fade = round(fade_seconds * sample_rate)
        if not 1 <= fade <= active // 2 or active >= period_samples:
            raise ValueError("淡变至少一个采样点，且不能超过开启时长的一半")
        if pump.size <= active:
            raise ValueError("音频太短，无法同时包含间歇开启和关闭区间")
        position = np.arange(pump.size) % period_samples
        mask = (position < active).astype(np.float64)
        attack = position < fade
        release = (position >= active - fade) & (position < active)
        mask[attack] *= 0.5 - 0.5 * np.cos(np.pi * position[attack] / fade)
        mask[release] *= 0.5 - 0.5 * np.cos(
            np.pi * (active - 1 - position[release]) / fade
        )
        parameters.update(
            {
                "period_samples": period_samples,
                "active_samples": active,
                "fade_samples": fade,
                "snr_reference": "active_mask_samples",
            }
        )
    selected = mask > 0
    if not selected.any():
        raise ValueError("间歇参数未产生有效开启区间")
    second = motor * mask
    first_level, second_level = rms(pump[selected]), rms(second[selected])
    if min(first_level, second_level) <= SILENCE_THRESHOLD:
        raise ValueError("开启区间包含静音，无法定义信噪比")
    gain = first_level / (second_level * 10 ** (snr_db / 20))
    parameters["motor_gain"] = gain
    return pump.copy(), second * gain, parameters


def peak_limit(audio: np.ndarray, ceiling: float) -> tuple[np.ndarray, float]:
    """以统一衰减限制峰值，返回波形及增益，保持两路相对能量和波形形状。"""
    if not 0 < ceiling < 1 or not np.all(np.isfinite(audio)):
        raise ValueError("峰值上限必须在 (0, 1) 内，波形必须为有限值")
    peak = float(np.max(np.abs(audio)))
    gain = min(1.0, ceiling / peak) if peak else 1.0
    return audio * gain, gain


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行路径、截取参数及融合参数；无效值通过 argparse 报错退出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pump", required=True, type=Path)
    parser.add_argument("--motor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--duration", type=float, default=None, help="默认取两路公共时长，上限 10 秒"
    )
    parser.add_argument("--pump-start", type=float, default=0)
    parser.add_argument("--motor-start", type=float, default=0)
    parser.add_argument(
        "--channel", type=int, default=0, help="默认首通道，-1 表示通道平均"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--snr-db", nargs="+", type=float, default=[-6, 0, 6])
    parser.add_argument("--period", type=float, default=2)
    parser.add_argument("--duty-cycle", type=float, default=0.5)
    parser.add_argument("--fade-ms", type=float, default=20)
    parser.add_argument("--peak", type=float, default=0.95)
    args = parser.parse_args(argv)
    values = [
        args.pump_start,
        args.motor_start,
        args.alpha,
        args.period,
        args.duty_cycle,
        args.fade_ms,
        args.peak,
        *args.snr_db,
    ]
    if args.duration is not None:
        values.append(args.duration)
    if not all(math.isfinite(value) for value in values):
        parser.error("数值参数必须为有限值")
    if (
        args.sample_rate < 1000
        or min(args.pump_start, args.motor_start) < 0
        or (args.duration is not None and args.duration <= 0)
    ):
        parser.error("采样率至少 1000 Hz，起点非负，时长为正")
    if not 0 < args.alpha < 1 or not 0 < args.peak < 1:
        parser.error("alpha 和 peak 必须在 (0, 1) 内")
    if (
        args.period <= 0
        or not 0 < args.duty_cycle < 1
        or args.fade_ms <= 0
        or any(abs(value) > 80 for value in args.snr_db)
    ):
        parser.error("请检查周期、占空比、淡变时长及 SNR 范围 [-80, 80]")
    return args


def run(args: argparse.Namespace) -> Path:
    """生成融合结果、参考输入及 JSON 清单，返回清单路径。

    先计算所有结果并检测文件冲突再写入；不覆盖既有文件。清单包含来源
    校验和、软件版本、截取位置、参数、统一防削波增益及量化后的幅度指标。
    """
    duration = args.duration
    if duration is None:
        duration = min(
            10.0,
            sf.info(args.pump).duration - args.pump_start,
            sf.info(args.motor).duration - args.motor_start,
        )
    pump, pump_info = load_audio(
        args.pump, args.sample_rate, args.pump_start, duration, args.channel
    )
    motor, motor_info = load_audio(
        args.motor, args.sample_rate, args.motor_start, duration, args.channel
    )
    length = min(round(duration * args.sample_rate), len(pump), len(motor))
    pump, motor = pump[:length], motor[:length]
    prepared = []
    for method in dict.fromkeys(args.methods):
        ratios = list(dict.fromkeys(args.snr_db)) if method == "snr" else [0.0]
        for ratio in ratios:
            first, second, parameters = mix_components(
                pump,
                motor,
                method,
                args.sample_rate,
                args.alpha,
                ratio,
                args.period,
                args.duty_cycle,
                args.fade_ms / 1000,
            )
            output, gain = peak_limit(first + second, args.peak)
            if rms(output) <= SILENCE_THRESHOLD:
                raise ValueError("融合结果近乎静音，请检查反相抵消或参数")
            name = f"snr_{ratio:+g}dB.wav" if method == "snr" else f"{method}.wav"
            prepared.append(
                (
                    name,
                    output,
                    {
                        "method": method,
                        "parameters": parameters,
                        "common_output_gain": gain,
                        "component_ratio_db": 20 * math.log10(rms(first) / rms(second)),
                        "synthetic": True,
                        "labels": ["pump_abnormal", "motor_abnormal"],
                    },
                )
            )
    for name, audio in (("source_pump.wav", pump), ("source_motor.wav", motor)):
        output, gain = peak_limit(audio, args.peak)
        prepared.append((name, output, {"reference": True, "output_gain": gain}))
    names = [name for name, _, _ in prepared] + ["manifest.json"]
    if len(set(names)) != len(names):
        raise ValueError("参数产生重复输出文件名，请增大 SNR 参数间隔")
    for name in names:
        if (args.output_dir / name).exists():
            raise FileExistsError(f"输出已存在，请选择新目录：{args.output_dir / name}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "sample_rate": args.sample_rate,
        "frames": length,
        "duration_seconds": length / args.sample_rate,
        "subtype": "PCM_16",
        "sources": {"pump": pump_info, "motor": motor_info},
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
        },
        "label_note": "标签依据用户指定的异常输入；脚本不自动诊断故障类型。",
        "outputs": [],
    }
    for name, output, record in prepared:
        path = args.output_dir / name
        sf.write(path, output, args.sample_rate, subtype="PCM_16")
        decoded, _ = sf.read(path, dtype="float64")
        record.update(
            {
                "file": name,
                "sha256": file_digest(path),
                "peak": float(np.max(np.abs(decoded))),
                "rms": rms(decoded),
            }
        )
        manifest["outputs"].append(record)
        print(f"已生成：{path}")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    """执行命令行融合流程，失败时给出中文原因并以非零状态退出。"""
    args = parse_args()
    try:
        manifest = run(args)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"融合失败：{error}") from error
    print(f"融合完成，参数清单：{manifest}")


if __name__ == "__main__":
    main()
