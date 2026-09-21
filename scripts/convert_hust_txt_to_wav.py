#!/usr/bin/env python3
"""按 HUSTmotor 数据说明提取 TXT 中的 Sound 通道并重采样为 WAV。

原始数据采样率为 25600 Hz；第 1 列为时间，第 2-4 列为振动，
第 5 列为声音。文件名中的 5/10/20/30 HZ 为转速工况，不是采样率。
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SOURCE_SAMPLE_RATE = 25600
DEFAULT_INPUT_DIR = Path(r"D:\dataset\HUST\Raw data")
DATA_MARKER = "Time (seconds) and Data Channels"
LOGGER = logging.getLogger(__name__)


def load_hust_sound(path: Path) -> np.ndarray:
    """读取 HUST TXT，逐块校验时间轴并按原始行顺序返回拼接后的声音通道。

    Args:
        path: 包含时间、三轴振动和 Sound 通道的 HUST TXT 文件路径。

    Returns:
        一维 float64 声音数组，保留原始幅值，不进行归一化或去均值。
        兼容整段连续时间与每块时间归零的导出格式，不按时间列重新排序。

    Raises:
        ValueError: 数据区标记、通道、行数、数值或 25600 Hz 时间轴不合法。
        OSError: 输入文件无法读取。
    """
    declared_rows = None
    declared_blocks = None
    sound_column = 4
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            text = line.strip()
            if text.startswith("Total Data Rows"):
                declared_rows = int(text.split()[-1])
            elif text.startswith("Number of Blocks"):
                declared_blocks = int(text.split()[-1])
                if declared_blocks < 1:
                    raise ValueError(f"数据块数量必须为正整数：{path}")
            elif text.startswith("Legend"):
                channels = text.split()[1:]
                if len(channels) != 4 or channels.count("Sound") != 1:
                    raise ValueError(f"通道声明不符合 HUST 格式：{path}")
                sound_column = channels.index("Sound") + 1
            elif text == DATA_MARKER:
                break
        else:
            raise ValueError(f"未找到 TXT 数据区标记：{path}")
        data = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != 5:
        raise ValueError(f"数据须至少包含两行、五列：{path}")
    if not np.isfinite(data).all():
        raise ValueError(f"数据包含 NaN 或 Inf：{path}")
    if declared_rows is not None and data.shape[0] != declared_rows:
        raise ValueError(f"实际行数与 Total Data Rows 不一致：{path}")
    # 实际 HUST 文件含多个数据块，每块时间归零；按原始行顺序拼接声音。
    reset_indices = np.flatnonzero(np.diff(data[:, 0]) <= 0) + 1
    boundaries = np.concatenate(([0], reset_indices, [data.shape[0]]))
    if reset_indices.size:
        if declared_blocks is not None and len(boundaries) - 1 != declared_blocks:
            raise ValueError(f"时间重置次数与 Number of Blocks 不一致：{path}")
        if not np.allclose(data[boundaries[:-1], 0], data[0, 0], rtol=0, atol=2e-8):
            raise ValueError(f"各数据块的时间起点不一致：{path}")
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        elapsed = data[start:end, 0] - data[start, 0]
        expected = np.arange(end - start, dtype=np.float64) / SOURCE_SAMPLE_RATE
        # TXT 时间保留八位小数；容忍舍入，拒绝块内丢行和错误采样率。
        if end - start < 2 or not np.allclose(elapsed, expected, rtol=0, atol=2e-8):
            raise ValueError(f"块内时间列与 25600 Hz 均匀采样不一致：{path}")
    return np.ascontiguousarray(data[:, sound_column])


def convert_file(
    source: Path, output: Path, sample_rate: int, overwrite: bool = False
) -> tuple[int, int]:
    """提取声音并抗混叠重采样，写入单声道 32 位浮点 WAV。

    Args:
        source: 原始 HUST TXT 文件路径。
        output: 输出 WAV 路径；自动创建父目录。
        sample_rate: 输出采样率，须为正整数，通常为 16000 Hz。
        overwrite: 是否允许覆盖已有 WAV，默认禁止覆盖。

    Returns:
        原始采样点数和输出采样点数。使用多相滤波保留录音时长，
        不做逐文件归一化；浮点 WAV 保留传感器幅值，避免 PCM 限幅。

    Raises:
        ValueError: 采样率、原始数据或重采样结果不合法。
        OSError: 文件读取或写入失败；禁止覆盖时同名输出会报错。
    """
    if type(sample_rate) is not int or sample_rate < 1:
        raise ValueError("输出采样率必须为正整数")
    signal = load_hust_sound(source)
    if sample_rate == SOURCE_SAMPLE_RATE:
        audio = signal
    else:
        divisor = math.gcd(SOURCE_SAMPLE_RATE, sample_rate)
        audio = resample_poly(
            signal, sample_rate // divisor, SOURCE_SAMPLE_RATE // divisor
        )
    expected_frames = (
        signal.size * sample_rate + SOURCE_SAMPLE_RATE - 1
    ) // SOURCE_SAMPLE_RATE
    if audio.size != expected_frames or not np.isfinite(audio).all():
        raise ValueError(f"重采样后的长度或数值异常：{source}")
    if np.max(np.abs(audio)) > np.finfo(np.float32).max:
        raise ValueError(f"音频幅值超出浮点 WAV 范围：{source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb" if overwrite else "xb") as stream:
        try:
            sf.write(stream, audio, sample_rate, format="WAV", subtype="FLOAT")
        except Exception:
            stream.close()
            output.unlink(missing_ok=True)
            raise
    return signal.size, audio.size


def main() -> int:
    """解析参数并递归转换 HUST TXT，记录中文日志并返回退出码。

    默认输入为 D:/dataset/HUST/Raw data，输出为输入目录同级的 wav_16000；
    自定义采样率时输出目录相应使用 wav_<采样率>。保留文件名与子目录结构。
    写入前检查全部输出路径，默认拒绝覆盖；单个数据文件转换失败后继续处理。
    全部成功返回 0，配置、路径或任一文件转换失败时返回 1。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir", nargs="?", type=Path, default=DEFAULT_INPUT_DIR,
        help="HUST TXT 根目录，默认 D:/dataset/HUST/Raw data",
    )
    parser.add_argument("--output-dir", type=Path, help="输出根目录")
    parser.add_argument("--sample-rate", type=int, default=16000, help="输出采样率，默认 16000 Hz")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 WAV")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s：%(message)s")
    try:
        if args.sample_rate < 1:
            raise ValueError("输出采样率必须为正整数")
        root = args.input_dir.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"输入目录不存在：{root}")
        output_root = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else root.parent / f"wav_{args.sample_rate}"
        )
        sources = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".txt"
        )
        if not sources:
            raise ValueError(f"没有找到 TXT 文件：{root}")
        jobs = [
            (source, output_root / source.relative_to(root).with_suffix(".wav"))
            for source in sources
        ]
        seen = set()
        for _, output in jobs:
            key = str(output).casefold()
            if key in seen:
                raise ValueError(f"多个输入文件映射到同一输出路径：{output}")
            seen.add(key)
            if output.exists() and (not args.overwrite or not output.is_file()):
                raise FileExistsError(f"输出已存在；确认需要覆盖后使用 --overwrite：{output}")
        failed = 0
        for index, (source, output) in enumerate(jobs, start=1):
            try:
                source_frames, output_frames = convert_file(
                    source, output, args.sample_rate, args.overwrite
                )
                LOGGER.info(
                    "[%d/%d] %s → %s；25600 → %d Hz；%d → %d 点；%.3f 秒",
                    index, len(jobs), source.name, output, args.sample_rate,
                    source_frames, output_frames, output_frames / args.sample_rate,
                )
            except (OSError, ValueError, RuntimeError) as error:
                failed += 1
                LOGGER.error("转换失败：%s；原因：%s", source, error)
        LOGGER.info(
            "转换完成：成功 %d，失败 %d；输出目录：%s",
            len(jobs) - failed, failed, output_root,
        )
        return 1 if failed else 0
    except (OSError, ValueError) as error:
        LOGGER.error("转换未完成：%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
