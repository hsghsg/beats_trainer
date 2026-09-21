#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 VL-RotMach 声学 MAT 批量转换为经过抗混叠重采样的 16 kHz 单声道 WAV。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import loadmat
from scipy.signal import resample_poly

LOGGER = logging.getLogger(__name__)
TARGET_RATE = 16000
DEFAULT_INPUT = Path(r"D:\dataset\韩国科学技术院多模态数据集\VL-RotMach\acoustic")


def load_acoustic(path: Path) -> tuple[np.ndarray, int, float]:
    """读取 Signal 结构，返回 Pa 声压数组、采样率及起始时间（秒）。

    从 x_values.increment 推导采样率，从 y_values.values 读取声压。
    校验单位、采样间隔、样本数和有限值；格式错误时抛出 ValueError。
    不修改原始文件，不对声压去均值或归一化。
    """
    try:
        signal = loadmat(path, simplify_cells=True)["Signal"]
        x_values = signal["x_values"]
        y_values = signal["y_values"]
        increment = float(x_values["increment"])
        start = float(x_values["start_value"])
        count = float(x_values["number_of_values"])
        time_unit = x_values["quantity"]["label"]
        pressure_unit = y_values["quantity"]["label"]
        raw = np.asarray(y_values["values"])
    except (KeyError, TypeError, ValueError, NotImplementedError) as error:
        raise ValueError(f"无法读取预期的 Signal 声学结构：{error}") from error
    if time_unit != "s" or pressure_unit != "Pa":
        raise ValueError(f"需要秒和 Pa 单位，实际为 {time_unit!r}、{pressure_unit!r}")
    if not math.isfinite(increment) or increment <= 0 or not math.isfinite(start):
        raise ValueError("采样间隔必须为有限正数，起始时间必须为有限值")
    rate_value = 1.0 / increment
    if not math.isfinite(rate_value):
        raise ValueError("采样间隔过小，无法计算采样率")
    rate = round(rate_value)
    if rate <= 0 or not math.isclose(rate_value, rate, rel_tol=1e-8):
        raise ValueError(f"不支持非整数采样率：{rate_value}")
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "fiu":
        raise ValueError("声压必须是至少包含两个样本的实数单声道向量")
    if count != raw.size:
        raise ValueError(f"元数据样本数 {count} 与实际样本数 {raw.size} 不一致")
    samples = raw.astype(np.float64, copy=False)
    if not np.isfinite(samples).all():
        raise ValueError("声压包含 NaN 或无穷值")
    return samples, rate, start


def resample_acoustic(samples: np.ndarray, source_rate: int) -> np.ndarray:
    """使用多相 FIR 抗混叠滤波重采样至 16 kHz，返回未缩放声压数组。

    51.2 kHz 对应上采样 5、下采样 16。输出长度为
    ceil(输入长度 × 16000 / 原采样率)，时长误差小于一个输出采样点。
    """
    divisor = math.gcd(source_rate, TARGET_RATE)
    return resample_poly(samples, TARGET_RATE // divisor, source_rate // divisor)


def convert_file(source: Path, destination: Path, subtype: str) -> dict:
    """转换单个 MAT 并原子写入 WAV，返回采样及幅值元数据。

    FLOAT 保留 Pa；PCM_16 将每文件重采样后峰值归一化至 0.99，静音不变。
    使用临时文件避免写入失败留下半成品；调用方负责覆盖策略。
    """
    samples, source_rate, start = load_acoustic(source)
    audio = resample_acoustic(samples, source_rate)
    if not np.isfinite(audio).all():
        raise ValueError("重采样结果包含非有限值")
    peak = float(np.max(np.abs(audio)))
    gain = 0.99 / peak if subtype == "PCM_16" and peak > 0 else 1.0
    if subtype == "FLOAT" and peak > np.finfo(np.float32).max:
        raise ValueError("声压超出 32 位浮点 WAV 数值范围")
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
        "source": str(source), "output": str(destination),
        "source_sample_rate": source_rate, "target_sample_rate": TARGET_RATE,
        "source_frames": int(samples.size), "output_frames": int(audio.size),
        "duration_seconds": audio.size / TARGET_RATE,
        "source_start_seconds": start, "source_unit": "Pa",
        "wav_subtype": subtype, "gain_applied": gain,
        "resampled_peak_pa": peak,
    }


def run(input_dir: Path, output_dir: Path, subtype: str, overwrite: bool) -> int:
    """递归转换 MAT、保留相对路径并生成本次运行的独立 JSON 清单。

    默认跳过已有 WAV；单文件失败时继续处理，存在失败返回 1，否则返回 0。
    无输入、无 MAT、输出在输入内部或编码错误时抛出 ValueError。
    """
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在：{input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("输出目录必须位于输入目录之外")
    if subtype not in ("FLOAT", "PCM_16"):
        raise ValueError("编码仅支持 FLOAT 或 PCM_16")
    sources = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mat"
    )
    if not sources:
        raise ValueError(f"未发现 MAT 文件：{input_dir}")
    records = []
    failures = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        destination = output_dir / source.relative_to(input_dir).with_suffix(".wav")
        if destination.exists() and not overwrite:
            LOGGER.info("跳过已有文件：%s", destination)
            records.append({
                "source": str(source), "output": str(destination), "status": "skipped",
            })
            continue
        try:
            record = convert_file(source, destination, subtype)
            record["status"] = "converted"
            records.append(record)
            LOGGER.info(
                "转换完成：%s，%d → %d Hz，%.3f 秒", source.name,
                record["source_sample_rate"], TARGET_RATE, record["duration_seconds"],
            )
        except (OSError, ValueError, TypeError, OverflowError) as error:
            failures += 1
            records.append({
                "source": str(source), "status": "failed", "error": str(error),
            })
            LOGGER.error("转换失败：%s：%s", source, error)
    descriptor, report = tempfile.mkstemp(
        prefix="conversion_", suffix=".json", dir=output_dir,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(records, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    converted = sum(record["status"] == "converted" for record in records)
    LOGGER.info(
        "处理结束：成功 %d，跳过 %d，失败 %d。清单：%s",
        converted, len(records) - converted - failures, failures, report,
    )
    return int(failures > 0)


def main() -> int:
    """解析命令行并启动批处理；参数、目录或转换错误时返回非零退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT, help="声学 MAT 目录")
    parser.add_argument("--output-dir", type=Path, help="默认是输入目录同级 acoustic_wav_16k")
    parser.add_argument(
        "--subtype", choices=("FLOAT", "PCM_16"), default="FLOAT",
        help="FLOAT 保留 Pa；PCM_16 按文件归一化，适合常规播放",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有同名 WAV")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s：%(message)s")
    output_dir = (
        args.output_dir or args.input_dir.expanduser().resolve().parent / "acoustic_wav_16k"
    )
    try:
        return run(args.input_dir, output_dir, args.subtype, args.overwrite)
    except (OSError, ValueError) as error:
        LOGGER.error("处理终止：%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

