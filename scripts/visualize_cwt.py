#!/usr/bin/env python3
"""读取 WAV 文件或递归扫描文件夹，调用 CWT 前处理器保存幅值时频图。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import librosa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beats_trainer.data.preprocessing import CWTAmplitudeTransform  # noqa: E402


def visualize_file(
    wav_path: Path,
    output_path: Path,
    voices_per_octave: int,
    duration: float | None,
) -> None:
    """读取单个 WAV 并调用前处理器保存 CWT 幅值时频图。

    Args:
        wav_path: 输入 WAV 文件路径。
        output_path: 输出 PNG 路径，父目录由前处理器自动创建。
        voices_per_octave: 每倍频程的尺度数，须为正整数。
        duration: 读取音频的秒数，None 表示读取完整音频。

    Returns:
        无返回值。保留原始采样率，多声道取平均。

    Raises:
        ValueError: 音频不足两个采样点或绘图参数非法。
        OSError: 音频读取或图片写入失败。
        ImportError: 缺少绘图依赖。
        RuntimeError: 音频解码或张量处理失败。
    """
    waveform, sample_rate = librosa.load(
        wav_path, sr=None, mono=True, duration=duration
    )
    if waveform.size < 2:
        raise ValueError("音频至少需要包含两个采样点")
    transform = CWTAmplitudeTransform(sample_rate, voices_per_octave)
    cwt = transform(torch.from_numpy(waveform))
    transform.visualize(cwt, output_path)


def main() -> int:
    """解析输入路径及绘图参数，处理单个 WAV 或递归处理文件夹。

    文件输入默认在音频旁保存同名 *_cwt.png；文件夹输入默认输出至
    输入目录上一级的 output_cwt，保留 WAV 相对于输入目录的子目录层级。
    --output 对单文件指定 PNG 路径，对文件夹指定输出根目录。
    批量处理时单个文件失败会记录中文日志并继续；全部成功返回 0，
    参数非法、未找到 WAV 或任一文件处理失败时返回 1。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="输入 WAV 文件或文件夹路径")
    parser.add_argument(
        "--output", type=Path,
        help="单文件输入时为 PNG 路径；文件夹输入时为输出根目录",
    )
    parser.add_argument("--duration", type=float, help="每个音频只读取前若干秒")
    parser.add_argument(
        "--voices-per-octave", type=int, default=12,
        help="每倍频程的尺度数，默认 12",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s：%(message)s")
    try:
        if args.voices_per_octave < 1:
            raise ValueError("每倍频程的尺度数必须为正整数")
        if args.duration is not None and not 0 < args.duration < float("inf"):
            raise ValueError("读取时长必须为有限正数")
        input_path = args.wav.expanduser().resolve()
        if input_path.is_dir():
            output_root = (
                args.output.expanduser().resolve()
                if args.output is not None
                else input_path.parent / "output_cwt"
            )
            if output_root.exists() and not output_root.is_dir():
                raise ValueError(f"输出根目录不能是已有文件：{output_root}")
            wav_paths = sorted(
                path for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() == ".wav"
            )
            if not wav_paths:
                raise ValueError(f"输入文件夹及子目录中没有 WAV 文件：{input_path}")
            jobs = [
                (
                    path,
                    output_root / path.relative_to(input_path).parent
                    / f"{path.stem}_cwt.png",
                )
                for path in wav_paths
            ]
        elif input_path.is_file() and input_path.suffix.lower() == ".wav":
            output = args.output or input_path.with_name(f"{input_path.stem}_cwt.png")
            jobs = [(input_path, output)]
        else:
            raise ValueError(f"输入必须为存在的 WAV 文件或文件夹：{input_path}")
    except (OSError, ValueError) as error:
        logging.error("CWT 可视化失败：%s", error)
        return 1

    failed_count = 0
    for index, (wav_path, output_path) in enumerate(jobs, start=1):
        logging.info("正在处理 [%d/%d]：%s", index, len(jobs), wav_path)
        try:
            visualize_file(
                wav_path, output_path, args.voices_per_octave, args.duration
            )
        except (OSError, ValueError, ImportError, RuntimeError) as error:
            failed_count += 1
            logging.error("CWT 可视化失败：%s；原因：%s", wav_path, error)
    logging.info(
        "CWT 处理完成：成功 %d 个，失败 %d 个",
        len(jobs) - failed_count, failed_count,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())
