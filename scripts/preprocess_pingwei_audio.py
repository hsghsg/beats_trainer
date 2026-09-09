#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据配套 YAML 筛选平圩设备录音、抗混叠重采样并生成定长训练音频。"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).with_suffix(".yaml")
INVALID_FILENAME_CHARACTERS = set('<>:"/\\|?*')


@dataclass(frozen=True)
class ProcessingConfig:
    """保存由 YAML 校验得到的路径、筛选、音频、命名和执行参数。"""

    input_dir: Path
    output_dir: Path
    extensions: list[str]
    filename_prefix: str
    filename_suffix: str
    recursive: bool
    clip_seconds: float
    target_sample_rate: int
    mono: bool
    wav_subtype: str
    tail_policy: str
    dataset: str
    collection_device: str
    target_device: str
    label: str
    filename_separator: str
    date_column: int
    time_column: int
    input_datetime_format: str
    output_datetime_format: str
    sequence_start: int
    sequence_width: int
    overwrite: bool
    dry_run: bool
    log_level: str

    @property
    def clip_frames(self) -> int:
        """返回目标采样率下每段的整数采样帧数；配置加载时保证结果有效。"""
        return round(self.clip_seconds * self.target_sample_rate)


@dataclass(frozen=True)
class SourcePlan:
    """记录单个源文件的音频元数据、转换后长度和无冲突的输出序号范围。"""

    path: Path
    sample_rate: int
    channels: int
    source_frames: int
    target_frames: int
    timestamp: str
    first_sequence: int
    clip_count: int


def validate_name_part(value: str, field_name: str) -> None:
    """校验非空文件名字段，拒绝 Windows 禁用字符及控制字符，无效时抛出 ValueError。"""
    if (
        not value.strip()
        or value.endswith((" ", "."))
        or any(char in INVALID_FILENAME_CHARACTERS or ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{field_name} 包含空值或不合法的文件名字符：{value!r}")


def load_config(config_path: Path) -> ProcessingConfig:
    """以 UTF-8 加载并校验 YAML，返回配置；相对路径以 YAML 所在目录为基准。

    缺失、多余、类型错误、无效音频编码及无法得到整数采样帧的参数均抛出
    ValueError；不创建目录，也不修改输入或输出文件。
    """
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8-sig") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("YAML 顶层必须是参数映射")
    try:
        config = ProcessingConfig(**values)
    except TypeError as error:
        raise ValueError(f"YAML 参数缺失或存在未知参数：{error}") from error
    for name in ("recursive", "mono", "overwrite", "dry_run"):
        if type(values[name]) is not bool:
            raise ValueError(f"{name} 必须为 true 或 false")
    for name in (
        "target_sample_rate",
        "date_column",
        "time_column",
        "sequence_start",
        "sequence_width",
    ):
        if type(values[name]) is not int or values[name] < 1:
            raise ValueError(f"{name} 必须为正整数，列号从 1 开始")
    string_fields = (
        "input_dir",
        "output_dir",
        "filename_prefix",
        "filename_suffix",
        "wav_subtype",
        "tail_policy",
        "dataset",
        "collection_device",
        "target_device",
        "label",
        "filename_separator",
        "input_datetime_format",
        "output_datetime_format",
        "log_level",
    )
    for name in string_fields:
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"{name} 必须为非空字符串")
    if (
        type(config.clip_seconds) not in (int, float)
        or not math.isfinite(config.clip_seconds)
        or config.clip_seconds <= 0
    ):
        raise ValueError("clip_seconds 必须为有限正数")
    frame_count = config.clip_seconds * config.target_sample_rate
    if (
        not math.isfinite(frame_count)
        or round(frame_count) < 1
        or not math.isclose(frame_count, round(frame_count), rel_tol=0, abs_tol=1e-7)
    ):
        raise ValueError("clip_seconds × target_sample_rate 必须为正整数")
    if not isinstance(config.extensions, list) or not config.extensions:
        raise ValueError("extensions 必须为非空扩展名列表，如 [wav]")
    extensions = []
    for extension in config.extensions:
        if not isinstance(extension, str) or not extension.lstrip(".").isalnum():
            raise ValueError(f"无效的扩展名：{extension!r}")
        extensions.append("." + extension.lstrip(".").lower())
    if config.tail_policy not in ("drop", "pad"):
        raise ValueError("tail_policy 只能为 drop（丢弃）或 pad（补零）")
    if not sf.check_format("WAV", config.wav_subtype):
        raise ValueError(f"WAV 不支持配置的编码：{config.wav_subtype}")
    if config.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ValueError("log_level 必须为 DEBUG、INFO、WARNING 或 ERROR")
    for name in ("dataset", "collection_device", "target_device", "label"):
        validate_name_part(values[name], name)
    for name in ("input_dir", "output_dir"):
        path = Path(values[name]).expanduser()
        values[name] = (config_path.parent / path).resolve()
    if values["input_dir"] == values["output_dir"]:
        raise ValueError("输入和输出目录不能相同")
    values["extensions"] = extensions
    return ProcessingConfig(**values)


def extract_timestamp(path: Path, config: ProcessingConfig) -> str:
    """按配置的一基列号读取文件主名的日期和时间，严格解析并返回输出时间字段。

    文件列数不足、日期无效或时间格式不完整时抛出含原文件名的 ValueError。
    时间始终来自原文件名，不随切片偏移变化。
    """
    columns = path.stem.split(config.filename_separator)
    try:
        raw = f"{columns[config.date_column - 1]} {columns[config.time_column - 1]}"
        timestamp = datetime.strptime(raw, config.input_datetime_format)
        if timestamp.strftime(config.input_datetime_format) != raw:
            raise ValueError("日期和时间字段不符合完整格式")
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"无法从原文件名解析日期和时间：{path.name}；{error}"
        ) from error
    result = timestamp.strftime(config.output_datetime_format)
    validate_name_part(result, "输出时间字段")
    return result


def output_path(config: ProcessingConfig, timestamp: str, sequence: int) -> Path:
    """使用配置的命名字段和补零序号生成目标 WAV 路径，不执行文件写入。"""
    name = (
        f"{config.dataset}-{config.collection_device}-{config.target_device}-"
        f"{timestamp}-{config.label}-{sequence:0{config.sequence_width}d}.wav"
    )
    return config.output_dir / name


def build_plan(config: ProcessingConfig) -> list[SourcePlan]:
    """预检全部匹配文件并返回稳定排序的切片计划，任何重名冲突均在写入前报错。

    扩展名不区分大小写，前后缀匹配不含扩展名的主名且区分大小写。
    相同输出时间字段的文件接续编号，其他时间字段从 sequence_start 重新编号。
    """
    if not config.input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{config.input_dir}")
    candidates = (
        config.input_dir.rglob("*") if config.recursive else config.input_dir.iterdir()
    )
    paths = sorted(
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in config.extensions
        and path.stem.startswith(config.filename_prefix)
        and path.stem.endswith(config.filename_suffix)
    )
    if not paths:
        raise ValueError(
            f"没有找到符合扩展名、前缀和后缀条件的文件：{config.input_dir}"
        )
    next_sequences: dict[str, int] = defaultdict(int)
    plans = []
    for path in paths:
        timestamp = extract_timestamp(path, config)
        info = sf.info(path)
        target_frames = (
            info.frames * config.target_sample_rate + info.samplerate - 1
        ) // info.samplerate
        clip_count, remainder = divmod(target_frames, config.clip_frames)
        if remainder and config.tail_policy == "pad":
            clip_count += 1
        first_sequence = config.sequence_start + next_sequences[timestamp.casefold()]
        next_sequences[timestamp.casefold()] += clip_count
        plan = SourcePlan(
            path,
            info.samplerate,
            info.channels,
            info.frames,
            target_frames,
            timestamp,
            first_sequence,
            clip_count,
        )
        for sequence in range(first_sequence, first_sequence + clip_count):
            target = output_path(config, timestamp, sequence)
            if target.exists() and (not config.overwrite or not target.is_file()):
                raise FileExistsError(f"输出路径已存在，未写入任何文件：{target}")
        plans.append(plan)
        LOGGER.info(
            "预检：%s，%d Hz，%d 声道，%.3f 秒，计划 %d 段，尾段 %.3f 秒（%s）",
            path.name,
            info.samplerate,
            info.channels,
            info.duration,
            clip_count,
            remainder / config.target_sample_rate,
            config.tail_policy,
        )
    return plans


def load_resampled_audio(plan: SourcePlan, config: ProcessingConfig) -> np.ndarray:
    """读取单个文件并按需平均声道，再对整条录音抗混叠重采样，返回帧×声道数组。

    整条录音重采样后切分可避免内部切片边界的滤波重启；内存仅保留当前文件。
    读取结果与预检元数据不一致或包含 NaN/Inf 时抛出 ValueError。
    """
    audio, source_rate = sf.read(plan.path, dtype="float32", always_2d=True)
    if source_rate != plan.sample_rate or audio.shape != (
        plan.source_frames,
        plan.channels,
    ):
        raise ValueError(f"预检后源文件发生变化，请重新运行：{plan.path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"音频包含 NaN 或 Inf：{plan.path}")
    if config.mono and audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    if source_rate != config.target_sample_rate:
        divisor = math.gcd(source_rate, config.target_sample_rate)
        audio = resample_poly(
            audio,
            config.target_sample_rate // divisor,
            source_rate // divisor,
            axis=0,
        )
    if len(audio) != plan.target_frames or not np.isfinite(audio).all():
        raise ValueError(f"重采样结果的帧数或数值无效：{plan.path}")
    if (
        config.wav_subtype.startswith("PCM")
        and audio.size
        and np.max(np.abs(audio)) >= 1
    ):
        LOGGER.warning(
            "%s 的幅值达到 PCM 范围边界，可能削波；可将 wav_subtype 改为 FLOAT",
            plan.path.name,
        )
    return audio


def write_clip(path: Path, clip: np.ndarray, config: ProcessingConfig) -> None:
    """写入指定编码的 WAV；默认独占创建文件，写入失败时删除本次未完成的片段。

    仅 overwrite=true 时允许替换同名文件，原始音频始终不修改。
    """
    with path.open("wb" if config.overwrite else "xb") as stream:
        try:
            sf.write(
                stream,
                clip,
                config.target_sample_rate,
                format="WAV",
                subtype=config.wav_subtype,
            )
        except BaseException:
            stream.close()
            path.unlink(missing_ok=True)
            raise


def run(config: ProcessingConfig) -> dict[str, int]:
    """预检并逐文件执行处理，返回源文件、计划切片和实际写入数量。

    dry_run=true 时仅读取文件头，绝不创建输出目录；drop 丢弃不足一段的尾部，
    pad 将末段补零至完整长度。任意文件失败即停止，保留此前成功生成的片段。
    """
    plans = build_plan(config)
    summary = {
        "source_files": len(plans),
        "planned_clips": sum(plan.clip_count for plan in plans),
        "written_clips": 0,
    }
    LOGGER.info(
        "匹配 %d 个文件，计划输出 %d 段，目录：%s",
        summary["source_files"],
        summary["planned_clips"],
        config.output_dir,
    )
    if config.dry_run:
        LOGGER.info("预览完成，未创建或写入任何输出文件")
        return summary
    if summary["planned_clips"]:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    for index, plan in enumerate(plans, start=1):
        if not plan.clip_count:
            LOGGER.warning("跳过空文件或不足一段的录音：%s", plan.path.name)
            continue
        audio = load_resampled_audio(plan, config)
        for clip_index in range(plan.clip_count):
            start = clip_index * config.clip_frames
            clip = audio[start : start + config.clip_frames]
            if len(clip) < config.clip_frames:
                clip = np.pad(clip, ((0, config.clip_frames - len(clip)), (0, 0)))
            target = output_path(
                config, plan.timestamp, plan.first_sequence + clip_index
            )
            write_clip(target, clip, config)
            summary["written_clips"] += 1
        del audio, clip
        LOGGER.info(
            "处理完成 [%d/%d]：%s → %d 段",
            index,
            len(plans),
            plan.path.name,
            plan.clip_count,
        )
    LOGGER.info(
        "全部完成：%d 个源文件，生成 %d 个 WAV",
        summary["source_files"],
        summary["written_clips"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """解析唯一的配置文件选项、配置中文日志并执行处理，成功返回 0，失败返回 1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML 配置路径，默认读取脚本旁的同名 YAML",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        config = load_config(args.config)
        logging.getLogger().setLevel(config.log_level)
        run(config)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        LOGGER.error("处理失败：%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
