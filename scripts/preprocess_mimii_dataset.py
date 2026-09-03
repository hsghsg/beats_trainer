#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a MIMII dataset for BEATs Trainer training."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")
CLASS_NAMES = ("normal", "abnormal")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class AudioSample:
    """Represent one source audio file and its derived training target data."""

    source_path: Path
    class_name: str
    target_name: str


def parse_args() -> argparse.Namespace:
    """Parse CLI options for input path, output path, split ratios, and copy mode."""
    parser = argparse.ArgumentParser(
        description="Prepare MIMII audio files as data_ready/train|val|test folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset") / "MIMII",
        help="MIMII root directory. Default: dataset/MIMII",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_ready"),
        help="Prepared output directory. Default: data_ready",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio. Default: 0.7",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio. Default: 0.15",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio. Default: 0.15",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits. Default: 42",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them. Default: copy files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target files when names conflict.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the split summary without copying or moving files.",
    )
    return parser.parse_args()


def validate_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Validate that split ratios are non-negative and add up to exactly one."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must not be negative")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train-ratio, val-ratio, and test-ratio must sum to 1")


def build_target_name(audio_path: Path, input_dir: Path) -> tuple[str, str]:
    """Build the target file name from MIMII folder levels and return its label."""
    relative_parts = audio_path.relative_to(input_dir).parts
    if len(relative_parts) < 4:
        raise ValueError(f"audio path is too shallow: {audio_path}")

    machine_type, machine_id, class_name = relative_parts[:3]
    if class_name not in CLASS_NAMES:
        raise ValueError(f"unsupported class folder: {class_name}, file: {audio_path}")

    target_name = f"{machine_type}-{machine_id}-{class_name}-{audio_path.name}"
    return target_name, class_name


def collect_audio_samples(input_dir: Path) -> list[AudioSample]:
    """Collect audio samples from machine/id/class folders under a MIMII root."""
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    samples: list[AudioSample] = []
    target_names: set[str] = set()

    for class_dir in sorted(input_dir.glob("*/*/*")):
        if not class_dir.is_dir() or class_dir.name not in CLASS_NAMES:
            continue

        for audio_path in sorted(class_dir.iterdir()):
            if not audio_path.is_file():
                continue
            if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            target_name, class_name = build_target_name(audio_path, input_dir)
            if target_name in target_names:
                raise ValueError(f"duplicate target file name: {target_name}")

            target_names.add(target_name)
            samples.append(
                AudioSample(
                    source_path=audio_path,
                    class_name=class_name,
                    target_name=target_name,
                )
            )

    if not samples:
        raise ValueError(f"no supported audio files found in: {input_dir}")

    return samples


def split_samples(
    samples: list[AudioSample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[AudioSample]]:
    """Split samples by class into train, validation, and test collections."""
    random_generator = random.Random(seed)
    splits = {split_name: [] for split_name in SPLIT_NAMES}

    for class_name in CLASS_NAMES:
        class_samples = [
            sample for sample in samples if sample.class_name == class_name
        ]
        random_generator.shuffle(class_samples)

        sample_count = len(class_samples)
        train_count = int(sample_count * train_ratio)
        val_count = int(sample_count * val_ratio)

        if sample_count > 0 and train_count == 0:
            train_count = 1
        if train_count + val_count > sample_count:
            val_count = max(0, sample_count - train_count)

        splits["train"].extend(class_samples[:train_count])
        splits["val"].extend(class_samples[train_count : train_count + val_count])
        splits["test"].extend(class_samples[train_count + val_count :])

    return splits


def prepare_output_dirs(output_dir: Path) -> None:
    """Create all train, validation, test, normal, and abnormal output folders."""
    for split_name in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            (output_dir / split_name / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )


def copy_or_move_sample(
    sample: AudioSample,
    target_path: Path,
    move_file: bool,
    overwrite: bool,
) -> None:
    """Copy or move a single audio sample while enforcing overwrite behavior."""
    if target_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"target exists, rerun with --overwrite if needed: {target_path}"
            )
        target_path.unlink()

    if move_file:
        shutil.move(str(sample.source_path), str(target_path))
        return

    shutil.copy2(sample.source_path, target_path)


def write_splits(
    splits: dict[str, list[AudioSample]],
    output_dir: Path,
    move_file: bool,
    overwrite: bool,
) -> None:
    """Write split samples to the BEATs Trainer directory layout."""
    prepare_output_dirs(output_dir)

    for split_name, split_samples_list in splits.items():
        for sample in split_samples_list:
            target_path = (
                output_dir / split_name / sample.class_name / sample.target_name
            )
            copy_or_move_sample(sample, target_path, move_file, overwrite)


def print_summary(splits: dict[str, list[AudioSample]]) -> None:
    """Print Chinese summary logs for generated split and class counts."""
    print("\u6570\u636e\u96c6\u6574\u7406\u7edf\u8ba1\uff1a")
    for split_name in SPLIT_NAMES:
        total_count = len(splits[split_name])
        normal_count = sum(
            1 for sample in splits[split_name] if sample.class_name == "normal"
        )
        abnormal_count = sum(
            1 for sample in splits[split_name] if sample.class_name == "abnormal"
        )
        print(
            f"  {split_name}: \u5171 {total_count} \u4e2a\u6587\u4ef6\uff0c"
            f"normal={normal_count}\uff0cabnormal={abnormal_count}"
        )


def main() -> None:
    """Run MIMII preprocessing from argument parsing through file output."""
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    print(f"\u5f00\u59cb\u6536\u96c6\u97f3\u9891\u6587\u4ef6\uff1a{args.input_dir}")
    samples = collect_audio_samples(args.input_dir)
    splits = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)

    if args.dry_run:
        print("\u9884\u89c8\u6a21\u5f0f\uff1a\u4e0d\u590d\u5236\u6216\u79fb\u52a8\u6587\u4ef6\u3002")
        print_summary(splits)
        return

    write_splits(splits, args.output_dir, args.move, args.overwrite)
    print(f"\u6570\u636e\u96c6\u5df2\u8f93\u51fa\u5230\uff1a{args.output_dir}")
    print_summary(splits)


if __name__ == "__main__":
    main()
