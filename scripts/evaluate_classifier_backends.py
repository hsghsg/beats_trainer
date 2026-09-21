#!/usr/bin/env python3
r"""在测试集上评估 BEATs 原生分类器和自定义 embedding 分类器后端。
python .\scripts\evaluate_classifier_backends.py `
  --recompute-features `
  --extract-batch-size 16 `
  --native-batch-size 16 `
  --output-dir artifacts/classifier_backend_evaluation_fresh
  """

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from beats_trainer import BEATsFeatureExtractor, Config  # noqa: E402
from beats_trainer.classifiers import (  # noqa: E402
    GMMCosineKNNHybridClassifier,
    LocalDensityKNNClassifier,
    RelativeMahalanobisDistanceClassifier,
)
from beats_trainer.core.model import BEATsLightningModule  # noqa: E402
from train_mimii_embedding_classifier import (  # noqa: E402
    apply_file_limit,
    collect_audio_files,
    extract_split_features,
)

DEFAULT_NATIVE_CHECKPOINT = (
    Path("logs")
    / "pump_pw_7fa13_finetune"
    / "version_3"
    / "checkpoints"
    / "last.ckpt"
)
DEFAULT_OUTPUT_DIR = Path("artifacts") / "classifier_backend_evaluation"


def parse_args() -> argparse.Namespace:
    """解析后端分类器测试集评估命令行参数。"""
    parser = argparse.ArgumentParser(
        description="评估 BEATs 原生分类器和 3 个自定义 embedding 分类器后端。"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_ready"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-dir", type=Path, help="新的测试集目录；指定后只重算测试特征。")
    parser.add_argument(
        "--train-feature-cache", type=Path,
        help="复用已有训练特征 NPZ，可指向其他评估输出目录；模型和训练数据须一致。",
    )
    parser.add_argument(
        "--native-checkpoint",
        type=Path,
        default=DEFAULT_NATIVE_CHECKPOINT,
        help="训练得到的 BEATs 原生分类器 checkpoint。",
    )
    parser.add_argument(
        "--embedding-model-path",
        type=Path,
        default=None,
        help="用于提取 embedding 的 checkpoint；默认复用 native checkpoint。",
    )
    parser.add_argument("--device", default="cuda", help="auto/cuda/cpu，默认 auto。")
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--native-batch-size", type=int, default=16)
    parser.add_argument("--positive-label", default="abnormal")
    parser.add_argument("--pauc-max-fpr", type=float, default=0.1)
    parser.add_argument("--max-files-per-split", type=int, default=None)
    parser.add_argument(
        "--recompute-features", "--recompute-test-features",
        dest="recompute_features", action="store_true",
        help="重新提取测试集特征，保留训练集缓存。",
    )
    parser.add_argument(
        "--recompute-train-features", action="store_true",
        help="显式重新提取训练集特征；更换特征模型或训练数据时使用。",
    )
    parser.add_argument("--skip-native", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path | None) -> Path | None:
    """将相对路径按项目根目录解析为绝对路径。"""
    if path is None:
        return None
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(device_name: str) -> torch.device:
    """根据用户参数选择 torch 运行设备。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_positive_index(classes: np.ndarray, positive_label: str) -> int:
    """返回正类标签在分类器 classes_ 中的列索引。"""
    matches = np.where(classes == positive_label)[0]
    if matches.size != 1:
        raise ValueError(f"无法定位正类标签 {positive_label!r}，类别列表：{classes}")
    return int(matches[0])


def validate_binary_labels(labels: np.ndarray, positive_label: str) -> np.ndarray:
    """校验标签是二分类，并返回按 numpy 稳定排序的类别数组。"""
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError(f"当前 AUC/pAUC 评估只支持二分类，实际类别：{classes}")
    if positive_label not in classes:
        raise ValueError(f"正类标签 {positive_label!r} 不在类别列表中：{classes}")
    return classes


def binary_targets(labels: np.ndarray, positive_label: str) -> np.ndarray:
    """将字符串标签转换为 sklearn ROC/AUC 使用的 0/1 标签。"""
    return (labels == positive_label).astype(np.int64)


def threshold_error_rates(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """按正类分数不低于阈值的规则计算 FAR、FRR，并线性插值估计 EER 交点。

    Args:
        y_true: 同时包含 0 和 1 的真实二分类标签，一维数组。
        scores: 与标签顺序一致的有限正类分数，一维数组。

    Returns:
        升序有限阈值、FAR=FPR、FRR=1-TPR，以及插值 EER 和阈值。
        离散分数下插值交点不保证可由单个实际阈值精确实现。

    Raises:
        ValueError: 输入形状、类别或分数不满足二分类计算条件。
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1 or y_true.shape != scores.shape:
        raise ValueError("标签与分数必须是一维且长度一致。")
    if not np.array_equal(np.unique(y_true), [0, 1]) or not np.isfinite(scores).all():
        raise ValueError("FAR/FRR 计算需要正负两类样本和有限分数。")
    far, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    # 使用高于最高分的有限阈值替换 sklearn 的无穷大起点，保留全拒绝端点。
    thresholds[0] = np.nextafter(scores.max(), np.inf)
    if not np.isfinite(thresholds[0]):
        raise ValueError("分数过大，无法生成有限的全拒绝阈值。")
    thresholds = thresholds[::-1]
    far = far[::-1]
    frr = (1.0 - tpr)[::-1]
    difference = far - frr
    right = int(np.flatnonzero(difference <= 0)[0])
    if difference[right] == 0:
        eer = float(far[right])
        eer_threshold = float(thresholds[right])
    else:
        left = right - 1
        weight = difference[left] / (difference[left] - difference[right])
        eer = float(far[left] + weight * (far[right] - far[left]))
        eer_threshold = float(thresholds[left] + weight * (thresholds[right] - thresholds[left]))
    return {
        "error_thresholds": thresholds,
        "far": far,
        "frr": frr,
        "EER": eer,
        "EER_threshold": eer_threshold,
    }


def evaluate_scores(
    backend: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    positive_label: str,
    pauc_max_fpr: float,
) -> dict[str, Any]:
    """计算分类指标、ROC、阈值 FAR/FRR 曲线及插值 EER 阈值。"""
    y_true = binary_targets(labels, positive_label)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return {
        "backend": backend,
        **threshold_error_rates(y_true, scores),
        "accuracy": float(accuracy_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=positive_label, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, scores)),
        "pAUC": float(roc_auc_score(y_true, scores, max_fpr=float(pauc_max_fpr))),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "predictions": predictions,
        "scores": scores,
    }


def ensure_checkpoint_config_paths(config: Config) -> None:
    """把 checkpoint 中保存的相对预训练模型路径修正为项目绝对路径。"""
    model_path = config.model.model_path
    if model_path is None:
        return
    checkpoint_path = Path(model_path)
    if checkpoint_path.is_absolute():
        return
    project_checkpoint_path = PROJECT_ROOT / checkpoint_path
    if project_checkpoint_path.exists():
        config.model.model_path = str(project_checkpoint_path)


def load_native_model(checkpoint_path: Path, device: torch.device) -> BEATsLightningModule:
    """从训练 checkpoint 加载 BEATs 原生分类器模型。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    config = hparams.get("config")
    if config is None:
        raise ValueError("checkpoint 中缺少 hyper_parameters.config，无法重建模型。")
    ensure_checkpoint_config_paths(config)
    num_classes = int(hparams.get("num_classes", config.model.num_classes))
    model = BEATsLightningModule(config, num_classes=num_classes)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing_keys:
        print(f"加载原生 checkpoint 时缺失参数：{missing_keys}")
    if unexpected_keys:
        print(f"加载原生 checkpoint 时发现额外参数：{unexpected_keys}")
    model.to(device)
    model.eval()
    return model


def load_audio_batch(
    audio_paths: list[Path],
    sample_rate: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取并补齐一批音频，返回 waveform 和 padding mask。"""
    audios: list[torch.Tensor] = []
    max_length = 0
    for audio_path in audio_paths:
        audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
        audios.append(audio_tensor)
        max_length = max(max_length, int(audio_tensor.shape[0]))

    padded: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for audio_tensor in audios:
        current_length = int(audio_tensor.shape[0])
        padding_length = max_length - current_length
        if padding_length > 0:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding_length))
        mask = torch.cat(
            [
                torch.zeros(current_length, dtype=torch.bool),
                torch.ones(padding_length, dtype=torch.bool),
            ]
        )
        padded.append(audio_tensor)
        masks.append(mask)
    return torch.stack(padded).to(device), torch.stack(masks).to(device)


def predict_native_probabilities(
    model: BEATsLightningModule,
    audio_paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """使用 BEATs 原生分类器对测试音频批量推理并返回概率。"""
    probability_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_start in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[batch_start : batch_start + batch_size]
            audio_batch, padding_mask = load_audio_batch(batch_paths, 16000, device)
            if not bool(padding_mask.any()):
                padding_mask = None
            logits = model(audio_batch, padding_mask)
            probabilities = torch.softmax(logits, dim=1)
            probability_batches.append(probabilities.detach().cpu().numpy())
            print(
                "BEATs 原生分类器推理进度："
                f"{min(batch_start + batch_size, len(audio_paths))}/{len(audio_paths)}"
            )
    return np.concatenate(probability_batches, axis=0)


def evaluate_native_backend(
    checkpoint_path: Path,
    test_dir: Path,
    classes: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """在测试集上评估 BEATs 原生 checkpoint 分类器。"""
    print(f"开始评估 BEATs 原生分类器：{checkpoint_path}")
    test_paths, labels = collect_audio_files(test_dir)
    test_paths, labels = apply_file_limit(test_paths, labels, args.max_files_per_split)
    model = load_native_model(checkpoint_path, device)
    probabilities = predict_native_probabilities(model, test_paths, args.native_batch_size, device)
    positive_index = get_positive_index(classes, args.positive_label)
    predictions = classes[np.argmax(probabilities, axis=1)]
    result = evaluate_scores(
        "beats-native-checkpoint",
        labels,
        predictions,
        probabilities[:, positive_index],
        args.positive_label,
        args.pauc_max_fpr,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def build_embedding_classifiers() -> dict[str, Any]:
    """创建三个自定义 embedding 后端分类器。"""
    return {
        "local-density-knn": LocalDensityKNNClassifier(k=15, density_k=20),
        "relative-mahalanobis": RelativeMahalanobisDistanceClassifier(
            covariance_type="diag",
            regularization=1e-4,
        ),
        "gmm-cosine-knn": GMMCosineKNNHybridClassifier(
            n_components=4,
            knn_k=15,
            alpha=0.6,
            random_state=42,
        ),
    }


def evaluate_embedding_backend(
    backend_name: str,
    classifier: Any,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """训练一个自定义 embedding 后端并在测试集上评估。"""
    print(f"开始训练后端分类器：{backend_name}")
    classifier.fit(train_features, train_labels)
    probabilities = classifier.predict_proba(test_features)
    predictions = classifier.predict(test_features)
    positive_index = get_positive_index(np.asarray(classifier.classes_), args.positive_label)
    result = evaluate_scores(
        backend_name,
        test_labels,
        predictions,
        probabilities[:, positive_index],
        args.positive_label,
        args.pauc_max_fpr,
    )
    print(
        f"{backend_name} 完成：accuracy={result['accuracy']:.4f}, "
        f"AUC={result['AUC']:.4f}, pAUC={result['pAUC']:.4f}"
    )
    return result


def metric_row(result: dict[str, Any]) -> dict[str, str]:
    """将一个后端的指标结果格式化为表格行。"""
    return {
        "backend": result["backend"],
        "accuracy": f"{result['accuracy']:.6f}",
        "recall": f"{result['recall']:.6f}",
        "f1": f"{result['f1']:.6f}",
        "AUC": f"{result['AUC']:.6f}",
        "pAUC": f"{result['pAUC']:.6f}",
        "EER": f"{result['EER']:.6f}",
        "EER_threshold": f"{result['EER_threshold']:.6f}",
    }


def metrics_table(results: list[dict[str, Any]]) -> str:
    """生成分类指标、AUC/pAUC、插值 EER 和对应阈值的 Markdown 表格。"""
    lines = [
        "| backend | accuracy | recall | f1 | AUC | pAUC | EER | EER_threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        row = metric_row(result)
        lines.append(
            "| {backend} | {accuracy} | {recall} | {f1} | {AUC} | {pAUC} | "
            "{EER} | {EER_threshold} |".format(**row)
        )
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    """递归转换 numpy 对象，保证评估结果可以写入 JSON。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def write_metrics_files(
    output_dir: Path,
    results: list[dict[str, Any]],
    positive_label: str,
    pauc_max_fpr: float,
) -> None:
    """输出 CSV、Markdown 和 JSON 三种评估结果文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backend", "accuracy", "recall", "f1", "AUC", "pAUC", "EER", "EER_threshold"]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_row(result) for result in results)

    markdown = "\n".join(
        [
            "# 后端分类器测试集评估结果",
            "",
            f"- 正类标签：`{positive_label}`",
            f"- 标准化 pAUC 最大 FPR：`{pauc_max_fpr}`",
            "- EER 与对应阈值由 FAR/FRR 曲线线性插值估计。",
            "",
            metrics_table(results),
            "",
        ]
    )
    (output_dir / "metrics.md").write_text(markdown, encoding="utf-8")

    payload = {
        "positive_label": positive_label,
        "pauc_max_fpr": pauc_max_fpr,
        "results": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_prediction_file(
    output_dir: Path,
    test_paths: list[str],
    test_labels: np.ndarray,
    results: list[dict[str, Any]],
) -> None:
    """输出每个测试样本在各后端上的预测标签和正类分数。"""
    fieldnames = ["path", "label"]
    for result in results:
        fieldnames.extend([f"{result['backend']}_prediction", f"{result['backend']}_score"])

    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (path, label) in enumerate(zip(test_paths, test_labels)):
            row = {"path": path, "label": label}
            for result in results:
                row[f"{result['backend']}_prediction"] = result["predictions"][index]
                row[f"{result['backend']}_score"] = f"{result['scores'][index]:.8f}"
            writer.writerow(row)


def svg_point(
    fpr: float,
    tpr: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[float, float]:
    """把 ROC 的 FPR/TPR 坐标转换为 SVG 像素坐标。"""
    return left + fpr * width, top + (1.0 - tpr) * height


def roc_points(
    fpr: np.ndarray,
    tpr: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
) -> str:
    """把 ROC 曲线数组转换为 SVG polyline points 字符串。"""
    points = [
        svg_point(float(x_value), float(y_value), left, top, width, height)
        for x_value, y_value in zip(fpr, tpr)
    ]
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def roc_axis_ticks(left: int, top: int, width: int, height: int) -> str:
    """根据绘图区的像素位置和尺寸，生成两轴 0.0～1.0、间隔 0.1 的 SVG 刻度线及标签。

    Args:
        left: 绘图区左边界的像素坐标。
        top: 绘图区上边界的像素坐标。
        width: 绘图区的像素宽度。
        height: 绘图区的像素高度。

    Returns:
        可嵌入 ROC SVG 的横纵轴刻度线和数值标签片段。
    """
    ticks: list[str] = []
    bottom = top + height
    for index in range(11):
        value = index / 10
        x, y = svg_point(value, value, left, top, width, height)
        ticks.extend(
            [
                f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" '
                f'y2="{bottom + 6}" stroke="#334155"/>',
                f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="12">{value:.1f}</text>',
                f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" '
                f'y2="{y:.2f}" stroke="#334155"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="Arial, sans-serif" font-size="12">{value:.1f}</text>',
            ]
        )
    return "\n  ".join(ticks)


def write_single_roc_svg(output_path: Path, result: dict[str, Any]) -> None:
    """为单个后端分类器绘制独立 ROC/AUC SVG 曲线。"""
    left, top, width, height = 82, 72, 560, 390
    start = svg_point(0.0, 0.0, left, top, width, height)
    end = svg_point(1.0, 1.0, left, top, width, height)
    points = roc_points(result["fpr"], result["tpr"], left, top, width, height)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="720" height="560" viewBox="0 0 720 560">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="360" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{result['backend']} ROC</text>
  <text x="360" y="58" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#334155">AUC={result['AUC']:.4f}, pAUC={result['pAUC']:.4f}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fafc" stroke="#334155" stroke-width="1.5"/>
  <line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="8 8"/>
  <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>
  <text x="{left + width / 2}" y="{top + height + 54}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15">False Positive Rate</text>
  <text x="24" y="{top + height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" transform="rotate(-90 24 {top + height / 2})">True Positive Rate</text>
  {roc_axis_ticks(left, top, width, height)}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def write_combined_roc_svg(output_path: Path, results: list[dict[str, Any]]) -> None:
    """把所有后端分类器的 ROC 曲线绘制在一张汇总 SVG 图中。"""
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    left, top, width, height = 82, 70, 560, 400
    start = svg_point(0.0, 0.0, left, top, width, height)
    end = svg_point(1.0, 1.0, left, top, width, height)
    curves: list[str] = []
    legends: list[str] = []
    for index, result in enumerate(results):
        color = colors[index % len(colors)]
        points = roc_points(result["fpr"], result["tpr"], left, top, width, height)
        curves.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_y = top + 24 + index * 42
        legends.append(
            f'<line x1="680" y1="{legend_y}" x2="720" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        legends.append(
            f'<text x="728" y="{legend_y + 4}" font-family="Arial, sans-serif" '
            f'font-size="13">{result["backend"]}</text>'
        )
        legends.append(
            f'<text x="728" y="{legend_y + 22}" font-family="Arial, sans-serif" '
            f'font-size="12" fill="#475569">AUC={result["AUC"]:.4f}, pAUC={result["pAUC"]:.4f}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="600" viewBox="0 0 980 600">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="490" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">后端分类器 ROC 对比</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fafc" stroke="#334155" stroke-width="1.5"/>
  <line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="8 8"/>
  {' '.join(curves)}
  {' '.join(legends)}
  {roc_axis_ticks(left, top, width, height)}
  <text x="{left + width / 2}" y="{top + height + 54}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15">False Positive Rate</text>
  <text x="24" y="{top + height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" transform="rotate(-90 24 {top + height / 2})">True Positive Rate</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def write_threshold_error_svg(output_path: Path, result: dict[str, Any]) -> None:
    """为一个分类器输出阈值 FAR/FRR SVG，标记插值 EER 交点及阈值辅助线。

    Args:
        output_path: SVG 输出路径，父目录不存在时自动创建。
        result: 包含 backend、error_thresholds、far、frr、EER、EER_threshold 的结果。

    Returns:
        无；将横轴为分数阈值、纵轴为错误率的曲线写入指定文件。
    """
    left, top, width, height = 82, 96, 640, 390
    thresholds = result["error_thresholds"]
    minimum = min(0.0, float(thresholds[0]))
    maximum = max(1.0, float(thresholds[-1]))
    span = maximum - minimum
    normalized = (thresholds - minimum) / span
    far_points = roc_points(normalized, result["far"], left, top, width, height)
    frr_points = roc_points(normalized, result["frr"], left, top, width, height)
    eer_x, eer_y = svg_point(
        (result["EER_threshold"] - minimum) / span, result["EER"], left, top, width, height,
    )
    ticks: list[str] = []
    bottom = top + height
    for index in range(11):
        fraction = index / 10
        x, y = svg_point(fraction, fraction, left, top, width, height)
        value = minimum + fraction * span
        ticks.extend([
            f'<path d="M {x:.2f} {bottom} v 6 M {left} {y:.2f} h -6" stroke="#334155"/>',
            f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle">{value:.2g}</text>',
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{fraction:.1f}</text>',
        ])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="820" height="600" viewBox="0 0 820 600">
  <rect width="100%" height="100%" fill="white"/>
  <g font-family="Arial, sans-serif" font-size="12" fill="#334155">
    <text x="410" y="30" text-anchor="middle" font-size="21">{result['backend']} FAR / FRR</text>
    <text x="410" y="55" text-anchor="middle" font-size="14">EER={result['EER']:.6f}; threshold={result['EER_threshold']:.6f} (linear interpolation)</text>
    <text x="290" y="79" fill="#2563eb">FAR (FPR)</text>
    <text x="460" y="79" fill="#dc2626">FRR (1 - TPR)</text>
    <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fafc" stroke="#334155"/>
    {' '.join(ticks)}
    <polyline id="far-curve" points="{far_points}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
    <polyline id="frr-curve" points="{frr_points}" fill="none" stroke="#dc2626" stroke-width="2.5"/>
    <path d="M {eer_x:.2f} {top} V {bottom} M {left} {eer_y:.2f} H {left + width}" stroke="#7c3aed" stroke-dasharray="5 5"/>
    <circle id="eer-point" cx="{eer_x:.2f}" cy="{eer_y:.2f}" r="5" fill="#7c3aed"/>
    <text x="{min(eer_x + 9, left + width - 34):.2f}" y="{max(top + 16, eer_y - 10):.2f}" fill="#7c3aed">EER</text>
    <text x="{left + width / 2}" y="{bottom + 54}" text-anchor="middle" font-size="15">Threshold (positive score &gt;= threshold)</text>
    <text x="24" y="{top + height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 24 {top + height / 2})">Error rate</text>
  </g>
</svg>
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def write_roc_files(output_dir: Path, results: list[dict[str, Any]]) -> None:
    """输出每个后端的 ROC、阈值 FAR/FRR 曲线及汇总 ROC 曲线。"""
    roc_dir = output_dir / "roc_curves"
    for result in results:
        write_single_roc_svg(roc_dir / f"{result['backend']}.svg", result)
        write_threshold_error_svg(
            output_dir / "threshold_curves" / f"{result['backend']}.svg", result,
        )
    write_combined_roc_svg(output_dir / "roc_curves_combined.svg", results)


def main() -> None:
    """执行测试集评估、绘制 ROC 曲线并保存指标表格。"""
    args = parse_args()
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    native_checkpoint = resolve_project_path(args.native_checkpoint)
    embedding_model_path = resolve_project_path(args.embedding_model_path) or native_checkpoint
    if data_dir is None or output_dir is None or embedding_model_path is None:
        raise ValueError("data-dir、output-dir、embedding checkpoint 都不能为空。")

    train_dir = data_dir / "train"
    test_dir = resolve_project_path(args.test_dir) or data_dir / "test"
    device = resolve_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels = collect_audio_files(train_dir)
    test_paths, test_labels = collect_audio_files(test_dir)
    train_paths, train_labels = apply_file_limit(train_paths, train_labels, args.max_files_per_split)
    test_paths, test_labels = apply_file_limit(test_paths, test_labels, args.max_files_per_split)
    classes = validate_binary_labels(np.concatenate([train_labels, test_labels]), args.positive_label)

    print(f"使用设备：{device}")
    print(f"类别顺序：{classes.tolist()}，正类：{args.positive_label}")
    print(f"训练集样本数：{len(train_labels)}，测试集样本数：{len(test_labels)}")
    print(f"embedding checkpoint：{embedding_model_path}")

    cache_tag = "all" if args.max_files_per_split is None else f"max{args.max_files_per_split}"
    train_cache = resolve_project_path(args.train_feature_cache)
    if train_cache is not None and not train_cache.exists() and not args.recompute_train_features:
        raise FileNotFoundError(f"指定的训练特征缓存不存在：{train_cache}")
    train_cache = train_cache or output_dir / "features" / f"train_features_{cache_tag}.npz"
    extractor = BEATsFeatureExtractor(model_path=embedding_model_path, device=str(device), pooling="mean")
    train_features, train_labels, _ = extract_split_features(
        extractor,
        "train",
        train_dir,
        train_cache,
        args.extract_batch_size,
        args.max_files_per_split,
        args.recompute_train_features,
    )
    test_features, test_labels, test_path_strings = extract_split_features(
        extractor,
        "test",
        test_dir,
        output_dir / "features" / f"test_features_{cache_tag}.npz",
        args.extract_batch_size,
        args.max_files_per_split,
        args.recompute_features or args.test_dir is not None,
    )
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []
    if not args.skip_native:
        if native_checkpoint is None or not native_checkpoint.exists():
            print(f"跳过 BEATs 原生分类器，checkpoint 不存在：{native_checkpoint}")
        else:
            results.append(evaluate_native_backend(native_checkpoint, test_dir, classes, args, device))

    for backend_name, classifier in build_embedding_classifiers().items():
        results.append(
            evaluate_embedding_backend(
                backend_name,
                classifier,
                train_features,
                train_labels,
                test_features,
                test_labels,
                args,
            )
        )

    write_metrics_files(output_dir, results, args.positive_label, args.pauc_max_fpr)
    write_prediction_file(output_dir, test_path_strings, test_labels, results)
    write_roc_files(output_dir, results)

    print("\n测试集指标：")
    print(metrics_table(results))
    print(f"\n指标 CSV：{output_dir / 'metrics.csv'}")
    print(f"指标 Markdown：{output_dir / 'metrics.md'}")
    print(f"独立 ROC 曲线目录：{output_dir / 'roc_curves'}")
    print(f"汇总 ROC 曲线：{output_dir / 'roc_curves_combined.svg'}")
    print(f"阈值 FAR/FRR 曲线目录：{output_dir / 'threshold_curves'}")


if __name__ == "__main__":
    main()
