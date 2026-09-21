#!/usr/bin/env python3
"""递归复制 CUMTB WAV，以工况、转速、状态和原文件名生成扁平输出文件名。"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "CUMTB_acoustic_16k"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "tofusion" / "target"
LOGGER = logging.getLogger(__name__)


def renamed_filename(source: Path, input_dir: Path) -> str:
    """根据 WAV 相对于输入根目录的路径生成目标文件名。

    Args:
        source: 输入根目录内的 WAV 文件路径。
        input_dir: 包含 Cond_<编号> 子目录的输入根目录。

    Returns:
        Cond_1/1rpm/Health/vibandacoustic/va (1).wav 转换为
        C1-1rpm_Health_va(1).wav；忽略状态目录之后的中间目录，
        删除原文件名中的所有空白字符，并统一扩展名为 .wav。

    Raises:
        ValueError: 文件不在输入根目录内，路径层级不足或工况目录格式错误。
    """
    parts = source.relative_to(input_dir).parts
    if len(parts) < 4:
        raise ValueError(f"路径须包含工况、转速、状态三级目录：{source}")
    condition = re.fullmatch(r"Cond_(\d+)", parts[0], flags=re.IGNORECASE)
    if condition is None:
        raise ValueError(f"工况目录须使用 Cond_<编号> 格式：{source}")
    filename = re.sub(r"\s+", "", source.stem) + ".wav"
    return f"C{condition.group(1)}-{parts[1]}_{parts[2]}_{filename}"


def build_copy_plan(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """扫描 WAV 并预检全部目标路径，返回按源路径排序的复制计划。

    Args:
        input_dir: 已解析为绝对路径的源目录，递归扫描且扩展名不区分大小写。
        output_dir: 已解析为绝对路径的输出目录，不得位于输入目录内。

    Returns:
        (源文件路径, 目标文件路径) 列表；仅预检，不创建目录或写入文件。

    Raises:
        ValueError: 输入目录无效、没有 WAV、路径格式错误或重命名产生冲突。
        FileExistsError: 输出路径已存在，或输出根路径不是目录。
        OSError: 目录扫描失败。
    """
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在：{input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("输出目录不能等于输入目录或位于其内部")
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"输出根路径不是目录：{output_dir}")
    sources = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not sources:
        raise ValueError(f"没有找到 WAV 文件：{input_dir}")
    plan = []
    seen: dict[str, Path] = {}
    for source in sources:
        target = output_dir / renamed_filename(source, input_dir)
        key = target.name.casefold()
        if key in seen:
            raise ValueError(f"重命名冲突：{seen[key]} 和 {source} → {target.name}")
        if target.exists():
            raise FileExistsError(f"目标已存在，未开始复制：{target}")
        seen[key] = source
        plan.append((source, target))
    return plan


def copy_audio(source: Path, target: Path) -> None:
    """原样复制音频内容并保留文件时间等元数据，禁止覆盖已有目标。

    Args:
        source: 待复制的 WAV 文件路径，不修改其内容或文件名。
        target: 目标文件路径，其父目录须已创建。

    Returns:
        无返回值；复制成功后目标音频字节内容与源文件相同。

    Raises:
        OSError: 文件读取、独占创建、写入或元数据复制失败。
        复制失败时删除本次创建的不完整目标，已有目标始终不覆盖。
    """
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
    """解析目录参数并预检、复制 WAV，成功返回 0，路径或复制错误返回 1。

    默认输入为项目 dataset/CUMTB_acoustic_16k，默认输出为
    dataset/tofusion/target；--dry-run 仅打印复制计划，不创建输出目录。
    输出全部放在目标目录内，不重采样、不重新编码、不修改源文件。
    重名在复制前报错；复制期间出错即停止，已成功复制的文件保留。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="包含 Cond_<编号> 的输入根目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录，默认 dataset/tofusion/target")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不复制文件")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s：%(message)s")
    try:
        input_dir = args.input_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        plan = build_copy_plan(input_dir, output_dir)
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        for index, (source, target) in enumerate(plan, start=1):
            if not args.dry_run:
                copy_audio(source, target)
            LOGGER.info("[%d/%d] %s → %s", index, len(plan), source, target.name)
        LOGGER.info(
            "%s完成：共 %d 个 WAV，目标目录：%s",
            "预览" if args.dry_run else "复制", len(plan), output_dir,
        )
        return 0
    except (OSError, ValueError) as error:
        LOGGER.error("复制重命名未完成：%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
