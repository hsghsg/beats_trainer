"""隔离模型依赖，验证评估脚本的缓存分流、错误率计算和 SVG 输出。"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script_functions(relative_path: str) -> dict[str, Any]:
    """读取脚本函数定义并注入轻量依赖，避免验证缓存和指标时初始化 Torch 模型。

    Args:
        relative_path: 相对于仓库根目录的待测脚本路径。

    Returns:
        包含实际脚本函数和所需标准库、NumPy、sklearn 依赖的独立命名空间。
    """
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = dict(globals())
    namespace.update(DEFAULT_OUTPUT_DIR=Path("artifacts/classifier_backend_evaluation"),
                     DEFAULT_NATIVE_CHECKPOINT=Path("model.ckpt"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


class ClassifierEvaluationTests(unittest.TestCase):
    """检查阈值指标、输出文件和训练/测试缓存控制的实际行为。"""

    def setUp(self) -> None:
        """为每个用例加载独立的评估函数命名空间，避免模拟对象交叉污染。"""
        self.module = load_script_functions("scripts/evaluate_classifier_backends.py")

    def test_error_rates_match_direct_threshold_decisions(self) -> None:
        """逐阈值对照真实布尔分类，验证 FAR/FRR 定义、重复分数和端点。"""
        labels = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.1, 0.5, 0.8, 1.0, 1.0])
        result = self.module["threshold_error_rates"](labels, scores)
        self.assertTrue(np.isfinite(result["error_thresholds"]).all())
        self.assertTrue((np.diff(result["error_thresholds"]) > 0).all())
        for threshold, far, frr in zip(result["error_thresholds"], result["far"], result["frr"]):
            predicted = scores >= threshold
            self.assertAlmostEqual(far, predicted[labels == 0].mean())
            self.assertAlmostEqual(frr, (~predicted[labels == 1]).mean())
        self.assertEqual(result["far"][0], 1.0)
        self.assertEqual(result["frr"][-1], 1.0)

    def test_eer_known_cases(self) -> None:
        """验证完美分类、反向分类、常数分数及非采样交点的 EER 数值。"""
        cases = [
            ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.0, 0.8),
            ([0, 0, 1, 1], [0.8, 0.9, 0.1, 0.2], 1.0, 0.8),
            ([0, 1], [0.5, 0.5], 0.5, 0.5),
            ([0, 1], [1.0, 1.0], 0.5, 1.0),
            ([0, 1], [0.0, 0.0], 0.5, 0.0),
            ([0, 1, 1], [0.8, 0.2, 0.9], 0.5, 0.85),
        ]
        for labels, scores, eer, threshold in cases:
            with self.subTest(scores=scores):
                result = self.module["threshold_error_rates"](np.array(labels), np.array(scores))
                self.assertAlmostEqual(result["EER"], eer)
                self.assertAlmostEqual(result["EER_threshold"], threshold)

    def test_error_rates_reject_invalid_input(self) -> None:
        """验证单类别、非有限分数及长度不一致时返回明确错误。"""
        for labels, scores in [([0, 0], [0.1, 0.2]), ([0, 1], [0.1, np.nan]), ([0, 1], [0.1])]:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                self.module["threshold_error_rates"](np.array(labels), np.array(scores))

    def test_curves_and_metric_files(self) -> None:
        """检查所有输出包含 EER 指标，阈值图包含两条曲线和有限位置的 EER 标记。"""
        labels = np.array(["normal", "normal", "abnormal", "abnormal"])
        result = self.module["evaluate_scores"](
            "demo", labels, labels, np.array([0.1, 0.4, 0.3, 0.9]), "abnormal", 0.1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.module["write_metrics_files"](directory, [result], "abnormal", 0.1)
            self.module["write_roc_files"](directory, [result])
            with (directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["EER"]), result["EER"])
            self.assertIn("EER_threshold", row)
            self.assertIn("EER_threshold", (directory / "metrics.md").read_text(encoding="utf-8"))
            payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("far", payload["results"][0])
            root = ET.parse(directory / "threshold_curves/demo.svg").getroot()
            namespace = {"svg": "http://www.w3.org/2000/svg"}
            self.assertEqual(len(root.findall(".//svg:polyline", namespace)), 2)
            marker = root.find(".//svg:circle[@id='eer-point']", namespace)
            self.assertIsNotNone(marker)
            self.assertTrue(82 <= float(marker.get("cx")) <= 722)
            self.assertTrue(96 <= float(marker.get("cy")) <= 486)
            self.assertIn("linear interpolation", "".join(root.itertext()))
            for path in [directory / "roc_curves/demo.svg", directory / "roc_curves_combined.svg"]:
                texts = [node.text for node in ET.parse(path).getroot().findall("svg:text", namespace)]
                self.assertEqual(texts.count("0.0"), 2)
                self.assertEqual(texts.count("1.0"), 2)

    def test_cli_recompute_switches_are_independent(self) -> None:
        """验证原重算开关及测试别名只影响测试，训练重算需要独立开关。"""
        for switch in ["--recompute-features", "--recompute-test-features"]:
            with patch.object(sys, "argv", ["evaluate", switch]):
                args = self.module["parse_args"]()
            self.assertTrue(args.recompute_features)
            self.assertFalse(args.recompute_train_features)
        with patch.object(sys, "argv", ["evaluate", "--recompute-train-features"]):
            args = self.module["parse_args"]()
        self.assertTrue(args.recompute_train_features)
        self.assertFalse(args.recompute_features)

    def test_new_test_set_reuses_training_cache(self) -> None:
        """运行主流程并读取真实 NPZ，确保新输出目录只提取测试集且显式开关可重算训练集。"""
        extraction = load_script_functions("scripts/train_mimii_embedding_classifier.py")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train_paths = [directory / "train/normal/a.wav", directory / "train/abnormal/b.wav"]
            test_paths = [directory / "new_test/normal/a.wav", directory / "new_test/abnormal/b.wav"]
            labels = np.array(["normal", "abnormal"])
            cache = directory / "old/features/train_features_all.npz"
            extraction["save_cached_features"](cache, np.array([[1.0], [2.0]]), labels, train_paths)
            collector = MagicMock(side_effect=[(train_paths, labels), (test_paths, labels), (test_paths, labels)])
            self.module["collect_audio_files"] = collector
            self.module["apply_file_limit"] = extraction["apply_file_limit"]
            extraction["collect_audio_files"] = collector
            extraction["tqdm"] = MagicMock()
            self.module["extract_split_features"] = extraction["extract_split_features"]
            extractor = MagicMock()
            extractor.extract_from_files.return_value = np.array([[3.0], [4.0]])
            self.module["BEATsFeatureExtractor"] = MagicMock(return_value=extractor)
            self.module["torch"] = MagicMock()
            self.module["torch"].cuda.is_available.return_value = False
            self.module["resolve_device"] = MagicMock(return_value="cpu")
            self.module["build_embedding_classifiers"] = MagicMock(return_value={})
            for name in ["write_metrics_files", "write_prediction_file", "write_roc_files"]:
                self.module[name] = MagicMock()
            argv = ["evaluate", "--data-dir", str(directory), "--test-dir", str(directory / "new_test"),
                    "--output-dir", str(directory / "new_output"), "--train-feature-cache", str(cache),
                    "--skip-native"]
            with patch.object(sys, "argv", argv):
                self.module["main"]()
            extractor.extract_from_files.assert_called_once_with(list(map(str, test_paths)), batch_size=16)
            with np.load(cache) as cached:
                np.testing.assert_array_equal(cached["features"], [[1.0], [2.0]])
            extractor.reset_mock()
            collector.side_effect = [(train_paths, labels), (test_paths, labels), (train_paths, labels), (test_paths, labels)]
            with patch.object(sys, "argv", argv + ["--recompute-train-features", "--recompute-features"]):
                self.module["main"]()
            self.assertEqual(extractor.extract_from_files.call_count, 2)


if __name__ == "__main__":
    unittest.main()
