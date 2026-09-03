#!/usr/bin/env python3
"""Train BEATs embedding classifiers on a prepared MIMII directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from beats_trainer import BEATsFeatureExtractor  # noqa: E402
from beats_trainer.classifiers import (  # noqa: E402
    BaseEmbeddingClassifier,
    GMMCosineKNNHybridClassifier,
    LocalDensityKNNClassifier,
    RelativeMahalanobisDistanceClassifier,
)


AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")
CLASSIFIER_CHOICES = (
    "local-density-knn",
    "relative-mahalanobis",
    "gmm-cosine-knn",
)


def parse_args() -> argparse.Namespace:
    """Parse command line options for embedding extraction and classifier training."""
    parser = argparse.ArgumentParser(
        description="Train a BEATs embedding classifier on data_ready train/val/test data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_ready"),
        help="Prepared data root. Default: data_ready",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "mimii_embedding_classifier",
        help="Output directory for features, metrics, and classifier.",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="local-density-knn",
        help="Embedding classifier backend. Default: local-density-knn",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional BEATs checkpoint path. Default: auto find or download.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device for BEATs embedding extraction. Default: auto",
    )
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction. Default: 16",
    )
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        default=None,
        help="Optional cap per split for smoke tests. Default: use all files.",
    )
    parser.add_argument(
        "--recompute-features",
        action="store_true",
        help="Recompute embeddings even when cached feature files exist.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="K value for KNN-based classifiers. Default: 15",
    )
    parser.add_argument(
        "--density-k",
        type=int,
        default=20,
        help="Neighbor count for local density estimation. Default: 20",
    )
    parser.add_argument(
        "--mahalanobis-covariance-type",
        choices=("diag", "full"),
        default="diag",
        help="Covariance type for Relative Mahalanobis. Default: diag",
    )
    parser.add_argument(
        "--mahalanobis-regularization",
        type=float,
        default=1e-4,
        help="Covariance regularization for Relative Mahalanobis. Default: 1e-4",
    )
    parser.add_argument(
        "--gmm-components",
        type=int,
        default=4,
        help="Maximum GMM components per class. Default: 4",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.6,
        help="GMM weight in the GMM-cosine-KNN hybrid. Default: 0.6",
    )
    return parser.parse_args()


def collect_audio_files(split_dir: Path) -> tuple[list[Path], np.ndarray]:
    """Collect direct child audio files from class folders in one split directory."""
    if not split_dir.exists():
        raise FileNotFoundError(f"split directory does not exist: {split_dir}")

    audio_paths: list[Path] = []
    labels: list[str] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for audio_path in sorted(class_dir.iterdir()):
            if not audio_path.is_file():
                continue
            if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            audio_paths.append(audio_path)
            labels.append(class_dir.name)

    if not audio_paths:
        raise ValueError(f"no audio files found in split directory: {split_dir}")

    return audio_paths, np.asarray(labels)


def apply_file_limit(
    audio_paths: list[Path],
    labels: np.ndarray,
    max_files: int | None,
) -> tuple[list[Path], np.ndarray]:
    """Limit a split to the first N records for quick smoke-test execution."""
    if max_files is None:
        return audio_paths, labels
    if max_files <= 0:
        raise ValueError("max-files-per-split must be positive when provided")
    return audio_paths[:max_files], labels[:max_files]


def load_cached_features(cache_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load cached embedding features, labels, and file paths from an NPZ file."""
    cached = np.load(cache_path, allow_pickle=False)
    features = cached["features"]
    labels = cached["labels"]
    paths = cached["paths"].tolist()
    return features, labels, paths


def save_cached_features(
    cache_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    paths: list[Path],
) -> None:
    """Persist embedding features, labels, and source paths to a compressed NPZ file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        paths=np.asarray([str(path) for path in paths]),
    )


def extract_split_features(
    extractor: BEATsFeatureExtractor,
    split_name: str,
    split_dir: Path,
    cache_path: Path,
    extract_batch_size: int,
    max_files: int | None,
    recompute_features: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load cached embeddings or extract BEATs embeddings for one data split."""
    if cache_path.exists() and not recompute_features:
        print(f"\u8bfb\u53d6\u7f13\u5b58\u7279\u5f81\uff1a{cache_path}")
        return load_cached_features(cache_path)

    audio_paths, labels = collect_audio_files(split_dir)
    audio_paths, labels = apply_file_limit(audio_paths, labels, max_files)
    print(
        f"\u5f00\u59cb\u63d0\u53d6 {split_name} \u7279\u5f81\uff0c"
        f"\u6837\u672c\u6570\uff1a{len(audio_paths)}"
    )
    features = extractor.extract_from_files(
        [str(path) for path in audio_paths],
        batch_size=extract_batch_size,
    )
    save_cached_features(cache_path, features, labels, audio_paths)
    return features, labels, [str(path) for path in audio_paths]


def build_classifier(args: argparse.Namespace) -> BaseEmbeddingClassifier:
    """Create the selected embedding classifier from parsed command line options."""
    if args.classifier == "local-density-knn":
        return LocalDensityKNNClassifier(k=args.knn_k, density_k=args.density_k)
    if args.classifier == "relative-mahalanobis":
        return RelativeMahalanobisDistanceClassifier(
            covariance_type=args.mahalanobis_covariance_type,
            regularization=args.mahalanobis_regularization,
        )
    if args.classifier == "gmm-cosine-knn":
        return GMMCosineKNNHybridClassifier(
            n_components=args.gmm_components,
            knn_k=args.knn_k,
            alpha=args.hybrid_alpha,
        )
    raise ValueError(f"unsupported classifier: {args.classifier}")


def evaluate_classifier(
    classifier: BaseEmbeddingClassifier,
    split_name: str,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a fitted classifier on one split and return serializable metrics."""
    predictions = classifier.predict(features)
    report = classification_report(
        labels,
        predictions,
        labels=classifier.classes_,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=classifier.classes_)
    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "labels": classifier.classes_.tolist(),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    print(
        f"{split_name}: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    return metrics


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalar and array values into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def save_metrics(metrics_path: Path, metrics: dict[str, Any]) -> None:
    """Write evaluation metrics to a UTF-8 JSON file with stable indentation."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as output_file:
        json.dump(to_jsonable(metrics), output_file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run BEATs embedding extraction, classifier fitting, evaluation, and saving."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\u521d\u59cb\u5316 BEATs \u7279\u5f81\u63d0\u53d6\u5668")
    extractor = BEATsFeatureExtractor(
        model_path=args.model_path,
        device=args.device,
        pooling="mean",
    )

    split_data = {}
    for split_name in ("train", "val", "test"):
        split_dir = args.data_dir / split_name
        cache_path = args.output_dir / f"{split_name}_features.npz"
        split_data[split_name] = extract_split_features(
            extractor=extractor,
            split_name=split_name,
            split_dir=split_dir,
            cache_path=cache_path,
            extract_batch_size=args.extract_batch_size,
            max_files=args.max_files_per_split,
            recompute_features=args.recompute_features,
        )

    classifier = build_classifier(args)
    train_features, train_labels, _ = split_data["train"]
    print(f"\u5f00\u59cb\u8bad\u7ec3\u5206\u7c7b\u5668\uff1a{args.classifier}")
    classifier.fit(train_features, train_labels)

    all_metrics = {
        "classifier": args.classifier,
        "classes": classifier.classes_.tolist(),
        "splits": {},
    }
    for split_name in ("train", "val", "test"):
        features, labels, _ = split_data[split_name]
        all_metrics["splits"][split_name] = evaluate_classifier(
            classifier,
            split_name,
            features,
            labels,
        )

    classifier_path = args.output_dir / f"{args.classifier}.pkl"
    metrics_path = args.output_dir / f"{args.classifier}_metrics.json"
    classifier.save(classifier_path)
    save_metrics(metrics_path, all_metrics)
    print(f"\u5206\u7c7b\u5668\u5df2\u4fdd\u5b58\uff1a{classifier_path}")
    print(f"\u8bc4\u4f30\u6307\u6807\u5df2\u4fdd\u5b58\uff1a{metrics_path}")


if __name__ == "__main__":
    main()
