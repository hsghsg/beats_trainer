#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 CUMTB 变桨轴承 CSV 的第六列声学数据批量重采样为 16 kHz WAV。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LOGGER = logging.getLogger(__name__)
SOURCE_RATE = 38500
TARGET_RATE = 16000
DEFAULT_INPUT = Path(r"D:\dataset\CUMTB风电变桨轴承")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "dataset" / "CUMTB_acoustic_16k"


def load_acoustic(path: Path) -> tuple[np.ndarray, int, int]:
    """读取无表头 CSV 的序号和第六列声音，返回声音数组、首尾样本序号。

    允许第七列为空（原文件末尾逗号），忽略第二至第五列振动通道。
    拒绝错误列结构、非有限声学值和非连续整数序号；不会把首行当表头丢弃。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        row = next(csv.reader(stream), [])
    if not (len(row) == 6 or (len(row) == 7 and not row[6].strip())):
        raise ValueError("CSV 必须包含六列数据，可附带末尾空列")
    data = np.loadtxt(
        path, delimiter=",", usecols=(0, 5), dtype=np.float64,
        ndmin=2, encoding="utf-8-sig",
    )
    if data.shape[0] < 2 or not np.isfinite(data).all():
        raise ValueError("至少需要两个样本，序号和声音不能包含 NaN 或无穷值")
    indices = data[:, 0]
    if (
        indices[0] < 0 or indices[-1] > 2**53
        or indices[0] != math.floor(indices[0])
        or not np.all(np.diff(indices) == 1)
    ):
        raise ValueError("第一列必须是连续递增的整数样本序号，不能作为秒时间戳")
    return data[:, 1], int(indices[0]), int(indices[-1])


def convert_file(source: Path, destination: Path, source_rate: int, subtype: str) -> dict:
    """读取单个 CSV、抗混叠重采样并原子写入 WAV，返回转换元数据。

    FLOAT 保留原声学数值，不推定物理单位；PCM_16 按文件将峰值缩放至 0.99。
    静音不缩放，不裁剪、不去均值。调用方负责处理已有输出和异常。
    """
    samples, first_index, last_index = load_acoustic(source)
    divisor = math.gcd(source_rate, TARGET_RATE)
    audio = resample_poly(samples, TARGET_RATE // divisor, source_rate // divisor)
    if not np.isfinite(audio).all():
        raise ValueError("重采样产生非有限数值")
    peak = float(np.max(np.abs(audio)))
    gain = 0.99 / peak if subtype == "PCM_16" and peak > 0 else 1.0
    if subtype == "FLOAT" and peak > np.finfo(np.float32).max:
        raise ValueError("声音数值超出浮点 WAV 范围")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(suffix=".wav", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        sf.write(temporary, audio * gain, TARGET_RATE, subtype=subtype, format="WAV")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source": str(source), "output": str(destination), "status": "converted",
        "source_sample_rate": source_rate, "target_sample_rate": TARGET_RATE,
        "source_frames": len(samples), "output_frames": len(audio),
        "source_first_index": first_index, "source_last_index": last_index,
        "source_duration_seconds": len(samples) / source_rate,
        "output_duration_seconds": len(audio) / TARGET_RATE,
        "acoustic_column_1based": 6, "wav_subtype": subtype,
        "gain_applied": gain, "resampled_peak": peak,
    }


def process_file(task: tuple[Path, Path, int, str, bool]) -> dict:
    """处理单文件任务，返回成功、跳过或失败记录，确保坏文件不终止整个批次。

    不覆盖时校验已有 WAV 的格式、声道、采样率及编码；不一致记为失败。
    一致则跳过，已有文件的源数据和源采样率不重新核验。
    """
    source, destination, source_rate, subtype, overwrite = task
    base = {"source": str(source), "output": str(destination)}
    try:
        if destination.exists() and not overwrite:
            info = sf.info(destination)
            if (
                info.samplerate != TARGET_RATE or info.channels != 1
                or info.subtype != subtype or info.format != "WAV" or info.frames < 1
            ):
                raise ValueError("已有 WAV 与要求不符，请使用新目录或 --overwrite")
            return {**base, "status": "skipped", "output_frames": info.frames}
        return convert_file(source, destination, source_rate, subtype)
    except (OSError, ValueError, RuntimeError, OverflowError) as error:
        return {**base, "status": "failed", "error": str(error)}


def run(
    input_dir: Path, output_dir: Path, source_rate: int = SOURCE_RATE,
    subtype: str = "FLOAT", overwrite: bool = False, workers: int = 4,
) -> int:
    """递归转换 CSV 并保留工况目录，将每文件结果即时写入独立 JSONL 清单。

    默认四个线程，每线程只加载一个 CSV；内存占用不随总数据量增长。
    已有输出默认跳过；存在失败返回 1，否则返回 0。无效参数抛出 ValueError。
    """
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在：{input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("输出目录必须位于输入目录之外")
    if type(source_rate) is not int or source_rate <= 0:
        raise ValueError("原始采样率必须为正整数，单位 Hz")
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("并行线程数必须在 1 到 32 之间")
    if subtype not in ("FLOAT", "PCM_16"):
        raise ValueError("编码仅支持 FLOAT 或 PCM_16")
    sources = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"
    )
    if not sources:
        raise ValueError(f"未发现 CSV：{input_dir}")
    tasks = [
        (p, output_dir / p.relative_to(input_dir).with_suffix(".wav"),
         source_rate, subtype, overwrite)
        for p in sources
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, report = tempfile.mkstemp(
        prefix="conversion_", suffix=".jsonl", dir=output_dir,
    )
    counts = {"converted": 0, "skipped": 0, "failed": 0}
    LOGGER.info("发现 %d 个 CSV，%d → %d Hz；输出：%s",
                len(sources), source_rate, TARGET_RATE, output_dir)
    LOGGER.info("本次转换清单：%s", report)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, record in enumerate(executor.map(process_file, tasks), 1):
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                counts[record["status"]] += 1
                if record["status"] == "failed":
                    LOGGER.error("转换失败：%s：%s", record["source"], record["error"])
                if index % 25 == 0 or index == len(tasks):
                    LOGGER.info(
                        "进度 %d/%d：成功 %d，跳过 %d，失败 %d",
                        index, len(tasks), counts["converted"], counts["skipped"], counts["failed"],
                    )
    LOGGER.info("处理结束，转换清单：%s", report)
    return int(counts["failed"] > 0)


def main() -> int:
    """解析输入输出、采样率、编码和并发参数，配置中文日志并返回批处理退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT, help="源 CSV 根目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="WAV 输出根目录")
    parser.add_argument("--source-rate", type=int, default=SOURCE_RATE, help="原采样率 Hz，默认 38500")
    parser.add_argument("--subtype", choices=("FLOAT", "PCM_16"), default="FLOAT",
                        help="FLOAT 保留原数值；PCM_16 逐文件归一化")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数，默认 4")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 WAV")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s：%(message)s")
    try:
        return run(
            args.input_dir, args.output_dir, args.source_rate,
            args.subtype, args.overwrite, args.workers,
        )
    except (OSError, ValueError) as error:
        LOGGER.error("处理终止：%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

