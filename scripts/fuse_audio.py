#!/usr/bin/env python3
"""将工厂背景噪声与故障音频按 +6、0、-6 dB 信噪比混合。"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff"}
SNR_LEVELS = (6, 0, -6)


def load_audio(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """读取音频并转换为单声道浮点波形，可按指定采样率重采样。

    Args:
        path: 输入音频文件路径，格式须由本机音频解码器支持。
        sample_rate: 目标采样率；None 表示保留文件原始采样率。

    Returns:
        一维 float64 波形和实际采样率。

    Raises:
        ValueError: 音频为空或包含非有限数值。
        OSError: 文件读取失败；解码器异常向调用方传播。
    """
    waveform, rate = librosa.load(path, sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"音频为空或包含非法数值：{path}")
    return waveform, rate


def align_noise(
    noise: np.ndarray, length: int, rng: np.random.Generator
) -> tuple[np.ndarray, int, bool]:
    """将非空背景噪声对齐至目标音频长度，不改变播放速度。

    Args:
        noise: 非空一维背景噪声波形。
        length: 所需采样点数量，须为正整数。
        rng: 控制截取起点的随机数生成器。

    Returns:
        对齐后的噪声、原噪声中的起始采样点、是否循环补齐。
        长噪声随机截取连续片段；短噪声从随机起点循环补齐。

    Raises:
        ValueError: 输入噪声为空或目标长度不是正数。
    """
    if noise.size == 0 or length < 1:
        raise ValueError("噪声和目标音频长度必须大于零")
    if noise.size >= length:
        start = int(rng.integers(noise.size - length + 1))
        return noise[start:start + length], start, False
    start = int(rng.integers(noise.size))
    shifted = np.concatenate((noise[start:], noise[:start]))
    return np.tile(shifted, math.ceil(length / noise.size))[:length], start, True


def mix_at_snr(
    signal: np.ndarray, noise: np.ndarray, snr_db: float
) -> tuple[np.ndarray, float, float]:
    """按整段波形平均功率设置信噪比，再统一缩放混合结果以防削波。

    Args:
        signal: 非空、有限的一维目标机器波形。
        noise: 与目标波形长度相同的有限一维背景噪声。
        snr_db: 目标信噪比，定义为 10*log10(目标功率/噪声功率)。

    Returns:
        混合波形、噪声增益、整体防削波增益。噪声增益为
        sqrt(mean(signal**2)/mean(noise**2))*10**(-snr_db/20)。
        整体增益同时作用于目标和噪声，峰值不超过 0.99，不改变信噪比。

    Raises:
        ValueError: 波形维度、长度、数值或功率不满足要求。
    """
    if signal.ndim != 1 or noise.shape != signal.shape or signal.size == 0:
        raise ValueError("目标和噪声必须为长度相同的非空一维波形")
    if not np.isfinite(signal).all() or not np.isfinite(noise).all():
        raise ValueError("目标或噪声包含非法数值")
    if not math.isfinite(snr_db):
        raise ValueError("信噪比必须为有限数值")
    signal_power = float(np.mean(signal ** 2))
    noise_power = float(np.mean(noise ** 2))
    if signal_power <= 0 or noise_power <= 0:
        raise ValueError("目标或选中的噪声片段为静音，无法设置有效信噪比")
    noise_gain = math.sqrt(signal_power / noise_power) * 10.0 ** (-snr_db / 20.0)
    mixed = signal + noise_gain * noise
    peak = float(np.max(np.abs(mixed)))
    output_gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0
    return mixed * output_gain, noise_gain, output_gain


def build_output_name(noise_path: Path, target_path: Path, snr_db: int) -> str:
    """保留完整噪声主名，并追加目标音频主名及信噪比生成 WAV 文件名。

    Args:
        noise_path: 背景噪声文件路径，原文件名中的序号保持不变。
        target_path: 目标音频路径，仅使用不含扩展名的文件主名。
        snr_db: 整数信噪比，当前使用 +6、0、-6 dB。

    Returns:
        形如 noise-0001-HUST_BF_10HZ(0dB).wav 的文件名；
        正信噪比显式带加号，零信噪比不带加号，不包含子目录。
    """
    snr_label = f"{snr_db:+d}" if snr_db != 0 else "0"
    return f"{noise_path.stem}-{target_path.stem}({snr_label}dB).wav"


def main() -> int:
    """解析目录参数，逐条配对背景噪声并生成三种信噪比的 WAV。

    --noise-dir 和 --target-dir 指定输入根目录下两个不同的直接子目录，默认 noise 和 target。
    递归扫描音频；每条背景噪声配一条目标音频，目标按随机打乱的轮次选取，
    默认种子为 42。同一配对的三个版本使用相同噪声片段。
    输出为单声道浮点 WAV，采样率由 --sample-rate 指定，默认 16000 Hz；
    目标和噪声均重采样至该采样率，输出时长以目标音频为准。
    保留噪声文件主名及序号，追加目标主名和信噪比，按信噪比分组写入 fusion。
    子目录为 snr_+6dB、snr_0dB、snr_-6dB；写入前校验本批输出路径冲突。
    fusion/manifest.csv 记录来源、输出相对路径及增益。
    再次运行会覆盖同路径输出。个别文件失败记录日志并继续。
    全部成功返回 0，输入无效或存在处理失败返回 1。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="包含噪声和故障音频子目录的根目录")
    parser.add_argument("--noise-dir", default="noise", help="背景噪声子目录名称，默认 noise")
    parser.add_argument("--target-dir", default="target", help="目标机器故障音频子目录名称，默认 target")
    parser.add_argument("--seed", type=int, default=42, help="非负随机种子，默认 42")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="混合及输出采样率，默认 16000 Hz")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s：%(message)s")
    try:
        root = args.input_dir.expanduser().resolve()
        noise_root = (root / args.noise_dir).resolve()
        target_root = (root / args.target_dir).resolve()
        output_root = root / "fusion"
        for folder in (noise_root, target_root):
            if not folder.is_dir() or folder.parent != root:
                raise ValueError(f"音频目录必须是输入根目录下存在的直接子目录：{folder}")
            if folder.name.lower() == "fusion":
                raise ValueError("fusion 为输出目录，不能用作音频输入目录")
        if noise_root == target_root:
            raise ValueError("背景噪声和故障音频必须使用不同目录")
        if args.sample_rate <= 0:
            raise ValueError("输出采样率必须为正整数")
        if args.seed < 0:
            raise ValueError("随机种子必须为非负整数")
        noise_files = sorted(
            p for p in noise_root.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
        )
        target_files = sorted(
            p for p in target_root.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
        )
        if not noise_files or not target_files:
            raise ValueError("背景噪声目录和故障音频目录均须包含可处理的音频文件")
        rng = np.random.default_rng(args.seed)
        target_order = rng.permutation(len(target_files))
        pairs = []
        seen_outputs = set()
        for index, noise_path in enumerate(noise_files):
            if index and index % len(target_files) == 0:
                target_order = rng.permutation(len(target_files))
            target_path = target_files[int(target_order[index % len(target_files)])]
            outputs = [
                output_root
                / (f"snr_{snr_db:+d}dB" if snr_db else "snr_0dB")
                / build_output_name(noise_path, target_path, snr_db)
                for snr_db in SNR_LEVELS
            ]
            for output in outputs:
                key = str(output.relative_to(output_root)).casefold()
                if key in seen_outputs:
                    raise ValueError(f"输出文件名冲突，尚未开始融合：{output}")
                seen_outputs.add(key)
            pairs.append((noise_path, target_path, outputs))
        output_root.mkdir(parents=True, exist_ok=True)
        failed_count = 0
        output_count = 0
        columns = [
            "target", "noise", "output", "snr_db", "sample_rate", "sample_count",
            "noise_start_sample", "noise_repeated", "noise_gain", "output_gain", "seed",
        ]
        with (output_root / "manifest.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index, (noise_path, target_path, outputs) in enumerate(pairs):
                try:
                    signal, sample_rate = load_audio(target_path, args.sample_rate)
                    noise, _ = load_audio(noise_path, sample_rate)
                    noise, start, repeated = align_noise(noise, signal.size, rng)
                    for snr_db, output in zip(SNR_LEVELS, outputs):
                        mixed, noise_gain, output_gain = mix_at_snr(signal, noise, snr_db)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(output, mixed, sample_rate, subtype="FLOAT")
                        writer.writerow({
                            "target": str(target_path.relative_to(root)),
                            "noise": str(noise_path.relative_to(root)),
                            "output": str(output.relative_to(root)),
                            "snr_db": snr_db, "sample_rate": sample_rate,
                            "sample_count": signal.size, "noise_start_sample": start,
                            "noise_repeated": repeated, "noise_gain": noise_gain,
                            "output_gain": output_gain, "seed": args.seed,
                        })
                        output_count += 1
                    logging.info(
                        "已处理 [%d/%d]：背景噪声 %s，目标音频 %s",
                        index + 1, len(noise_files), noise_path, target_path,
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    failed_count += 1
                    logging.error(
                        "融合失败：背景噪声 %s，目标音频 %s；原因：%s",
                        noise_path, target_path, error,
                    )
        logging.info(
            "融合完成：输出 %d 个音频，失败 %d 组；输出目录：%s",
            output_count, failed_count, output_root,
        )
        return 1 if failed_count else 0
    except (OSError, ValueError) as error:
        logging.error("声音融合未完成：%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
