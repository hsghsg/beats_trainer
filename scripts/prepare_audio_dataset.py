#!/usr/bin/env python3
"""将 normal、abnormal WAV 分别按 7:2:1 随机划分并复制到训练、验证和测试目录。"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMAL_DIR = PROJECT_ROOT / "dataset" / "tofusion" / "noise"
DEFAULT_ABNORMAL_DIR = PROJECT_ROOT / "dataset" / "tofusion" / "fusion" / "snr_0dB"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_ready"
SPLITS = ("train", "val", "test")
LOGGER = logging.getLogger(__name__)


def plan_class_split(
    source_dir: Path,
    output_dir: Path,
    label: str,
    rng: random.Random,
) -> list[tuple[Path, Path, str, str]]:
    """递归扫描单个类别的 WAV，随机打乱后生成 7:2:1 复制计划。

    Args:
        source_dir: 单个类别的输入根目录。
        output_dir: 划分后的数据集根目录。
        label: 类别名称，仅接受 normal 或 abnormal。
        rng: 已设置种子的随机数生成器；相同文件列表和种子可复现划分。

    Returns:
        (源路径, 输出路径, 数据划分, 类别) 列表。
        train 数量取总数的 70% 向下取整，val 取 20% 向下取整，
        剩余文件全部进入 test，确保每个文件恰好分配一次。
        输出路径保留源目录内部层级，避免不同子目录的同名文件冲突。

    Raises:
        ValueError: 类别无效、输入不存在或没有 WAV。
        OSError: 目录扫描失败。
    """
    if label not in ("normal", "abnormal"):
        raise ValueError(f"不支持的类别：{label}")
    if not source_dir.is_dir():
        raise ValueError(f"{label} 输入目录不存在：{source_dir}")
    files = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not files:
        raise ValueError(f"{label} 输入目录中没有 WAV：{source_dir}")
    rng.shuffle(files)
    train_count = len(files) * 7 // 10
    val_count = len(files) * 2 // 10
    boundaries = (train_count, train_count + val_count)
    plan = []
    for index, source in enumerate(files):
        split = (
            "train" if index < boundaries[0]
            else "val" if index < boundaries[1]
            else "test"
        )
        target = output_dir / split / label / source.relative_to(source_dir)
        plan.append((source, target, split, label))
    LOGGER.info(
        "%s：共 %d 个；train=%d，val=%d，test=%d",
        label, len(files), train_count, val_count,
        len(files) - train_count - val_count,
    )
    return plan


def copy_audio(source: Path, target: Path) -> None:
    """复制 WAV 内容及文件元数据，不修改源文件或覆盖已有目标。

    Args:
        source: 源 WAV 文件路径。
        target: 目标路径；缺失的父目录自动创建。

    Returns:
        无返回值，复制成功后文件内容保持不变。

    Raises:
        OSError: 文件读取、创建、写入或元数据复制失败。
        仅删除本次创建的不完整目标，已经存在的文件不会被覆盖。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream:
        output_stream = target.open("xb")
        try:
            with output_stream:
                shutil.copyfileobj(input_stream, output_stream)
            shutil.copystat(source, target)
        except BaseException:
            target.unlink(missing_ok=True)
            raise


def main() -> int:
    """解析参数、预检并复制两类 WAV，成功返回 0，参数或文件错误返回 1。

    默认 normal 来源为 dataset/tofusion/noise，abnormal 来源为
    dataset/tofusion/fusion/snr_0dB，输出为 data_ready/{train,val,test}/{类别}。
    两类分别按文件随机划分，不按同源录音分组。默认种子为 42。
    --dry-run 仅打印数量，不创建目录。目标类别目录必须为空，
    并禁止覆盖已有 split_manifest.csv，以防重复运行混入旧划分。
    实际复制时保存 split_manifest.csv，记录每条已成功复制音频的去向。
    复制失败即停止并保留已经成功写入的文件和对应清单记录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", type=Path, default=DEFAULT_NORMAL_DIR,
                        help="normal WAV 根目录")
    parser.add_argument("--abnormal-dir", type=Path, default=DEFAULT_ABNORMAL_DIR,
                        help="abnormal WAV 根目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="数据集输出根目录，默认 data_ready")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--dry-run", action="store_true", help="仅预览各类划分数量")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s：%(message)s")
    try:
        normal_dir = args.normal_dir.expanduser().resolve()
        abnormal_dir = args.abnormal_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        if (
            normal_dir == abnormal_dir
            or normal_dir in abnormal_dir.parents
            or abnormal_dir in normal_dir.parents
        ):
            raise ValueError("normal 和 abnormal 输入目录不能相同或互相包含")
        for source_dir in (normal_dir, abnormal_dir):
            if (
                source_dir == output_dir
                or source_dir in output_dir.parents
                or output_dir in source_dir.parents
            ):
                raise ValueError("输入与输出目录不能相同或互相包含")
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"输出根路径不是目录：{output_dir}")
        if args.seed < 0:
            raise ValueError("随机种子必须为非负整数")
        manifest_path = output_dir / "split_manifest.csv"
        if manifest_path.exists():
            raise FileExistsError(f"输出清单已存在，请使用新的空输出目录：{manifest_path}")
        for split in SPLITS:
            split_dir = output_dir / split
            if split_dir.exists() and not split_dir.is_dir():
                raise FileExistsError(f"划分路径不是目录：{split_dir}")
            for label in ("normal", "abnormal"):
                folder = split_dir / label
                if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
                    raise FileExistsError(f"目标类别目录不是空目录，未开始复制：{folder}")
        rng = random.Random(args.seed)
        plan = plan_class_split(normal_dir, output_dir, "normal", rng)
        plan.extend(plan_class_split(abnormal_dir, output_dir, "abnormal", rng))
        if args.dry_run:
            LOGGER.info("预览完成：总计 %d 个文件，未写入数据；输出目录：%s", len(plan), output_dir)
            return 0
        for split in SPLITS:
            for label in ("normal", "abnormal"):
                (output_dir / split / label).mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["split", "label", "source", "output", "seed"]
            )
            writer.writeheader()
            for index, (source, target, split, label) in enumerate(plan, start=1):
                copy_audio(source, target)
                writer.writerow({
                    "split": split, "label": label, "source": str(source),
                    "output": str(target.relative_to(output_dir)), "seed": args.seed,
                })
                if index == 1 or index % 1000 == 0 or index == len(plan):
                    LOGGER.info("已复制 %d/%d 个文件", index, len(plan))
        LOGGER.info("数据集准备完成：%s；划分清单：%s", output_dir, manifest_path)
        return 0
    except (OSError, ValueError) as error:
        LOGGER.error("数据集准备失败：%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
