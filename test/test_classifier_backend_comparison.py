"""比较 BEATs 原生分类头和自定义 embedding 分类器的后端差异。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn

from beats_trainer.classifiers import (
    GMMCosineKNNHybridClassifier,
    LocalDensityKNNClassifier,
    RelativeMahalanobisDistanceClassifier,
)


class ClassifierBackend(Protocol):
    """定义测试中所有分类器后端需要满足的最小推理协议。"""

    classes_: np.ndarray | None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ClassifierBackend":
        """使用训练 embedding 和标签拟合分类器后端。"""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """根据输入 embedding 输出每个样本的预测标签。"""

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """根据输入 embedding 输出每个样本在各类别上的概率分布。"""


@dataclass(frozen=True)
class BackendResult:
    """保存单个分类器后端在同一批探测样本上的预测结果。"""

    name: str
    classes: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


class BEATsNativeLinearClassifier:
    """用轻量线性分类头模拟 BEATs 原生分类器后端。"""

    def __init__(
        self,
        input_dim: int,
        classes: np.ndarray,
        learning_rate: float = 0.08,
        max_epochs: int = 150,
        seed: int = 7,
    ) -> None:
        """初始化 BEATs 风格线性分类头及其全量批训练参数。"""
        self.classes_ = np.asarray(classes)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = nn.Linear(input_dim, len(self.classes_))

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> "BEATsNativeLinearClassifier":
        """在已提取的 BEATs embedding 上训练原生线性分类头。"""
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        target_tensor = torch.as_tensor(self._labels_to_indices(labels), dtype=torch.long)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        self.model.train()
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            loss = loss_function(self.model(feature_tensor), target_tensor)
            loss.backward()
            optimizer.step()

        self.model.eval()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """输出 BEATs 原生线性分类头在输入 embedding 上的类别预测。"""
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """输出 BEATs 原生线性分类头的 softmax 类别概率。"""
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(feature_tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def _labels_to_indices(self, labels: np.ndarray) -> np.ndarray:
        """将字符串标签按 classes_ 顺序转换为交叉熵训练所需的整数标签。"""
        class_to_index = {
            class_label: class_index
            for class_index, class_label in enumerate(self.classes_)
        }
        return np.asarray([class_to_index[label] for label in labels], dtype=np.int64)


def make_multimodal_embedding_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成能暴露线性分类头与局部/概率型分类器差异的 embedding 数据。"""
    rng = np.random.default_rng(17)
    normal_main = np.array([0.0, 0.0, 0.0, 0.0]) + rng.normal(
        0.0,
        0.18,
        size=(36, 4),
    )
    normal_minor = np.array([4.0, 4.0, 0.0, 0.0]) + rng.normal(
        0.0,
        0.12,
        size=(6, 4),
    )
    abnormal = np.array([2.2, 2.2, 0.0, 0.0]) + rng.normal(
        0.0,
        0.16,
        size=(36, 4),
    )

    features = np.vstack([normal_main, normal_minor, abnormal])
    labels = np.asarray(
        ["normal"] * (len(normal_main) + len(normal_minor))
        + ["abnormal"] * len(abnormal)
    )
    probe_features = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [4.0, 4.0, 0.0, 0.0],
            [2.2, 2.2, 0.0, 0.0],
            [3.6, 3.6, 0.0, 0.0],
            [1.2, 1.2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return features, labels, probe_features


def build_backends(input_dim: int, classes: np.ndarray) -> dict[str, ClassifierBackend]:
    """创建 BEATs 原生分类头和三个自定义 embedding 分类器后端。"""
    return {
        "beats-native-linear": BEATsNativeLinearClassifier(
            input_dim=input_dim,
            classes=classes,
        ),
        "local-density-knn": LocalDensityKNNClassifier(
            k=5,
            density_k=5,
            metric="euclidean",
        ),
        "relative-mahalanobis": RelativeMahalanobisDistanceClassifier(
            covariance_type="full",
            regularization=1e-3,
        ),
        "gmm-cosine-knn": GMMCosineKNNHybridClassifier(
            n_components=2,
            knn_k=5,
            alpha=0.65,
            random_state=17,
        ),
    }


def collect_backend_results(
    features: np.ndarray,
    labels: np.ndarray,
    probe_features: np.ndarray,
) -> list[BackendResult]:
    """训练全部分类器后端，并收集它们在探测样本上的预测结果。"""
    classes = np.unique(labels)
    results: list[BackendResult] = []
    for name, backend in build_backends(features.shape[1], classes).items():
        backend.fit(features, labels)
        probabilities = backend.predict_proba(probe_features)
        predictions = backend.predict(probe_features)
        results.append(
            BackendResult(
                name=name,
                classes=np.asarray(backend.classes_),
                predictions=predictions,
                probabilities=probabilities,
            )
        )
    return results


def build_pairwise_mismatch_rates(
    results: list[BackendResult],
) -> dict[tuple[str, str], float]:
    """计算所有分类器后端两两之间的预测标签不一致率。"""
    mismatch_rates: dict[tuple[str, str], float] = {}
    for left_result, right_result in combinations(results, 2):
        mismatch_rates[(left_result.name, right_result.name)] = float(
            np.mean(left_result.predictions != right_result.predictions)
        )
    return mismatch_rates


def assert_backend_result_valid(
    result: BackendResult,
    expected_classes: np.ndarray,
    expected_sample_count: int,
) -> None:
    """校验单个后端输出的类别顺序、概率形状和概率归一化结果。"""
    expected_shape = (expected_sample_count, len(expected_classes))
    np.testing.assert_array_equal(result.classes, expected_classes)
    assert result.probabilities.shape == expected_shape
    assert result.predictions.shape == (expected_sample_count,)
    assert set(result.predictions).issubset(set(expected_classes))
    assert np.isfinite(result.probabilities).all()
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_backends_return_comparable_outputs() -> None:
    """验证四种分类器后端都能输出可比较的标签和概率结果。"""
    features, labels, probe_features = make_multimodal_embedding_case()
    expected_classes = np.unique(labels)

    results = collect_backend_results(features, labels, probe_features)

    assert {result.name for result in results} == {
        "beats-native-linear",
        "local-density-knn",
        "relative-mahalanobis",
        "gmm-cosine-knn",
    }
    for result in results:
        assert_backend_result_valid(result, expected_classes, len(probe_features))


def test_classifier_backends_show_prediction_differences() -> None:
    """验证 BEATs 原生分类头和三个自定义后端存在可观测预测差异。"""
    features, labels, probe_features = make_multimodal_embedding_case()

    results = collect_backend_results(features, labels, probe_features)
    mismatch_rates = build_pairwise_mismatch_rates(results)

    native_mismatch_rates = [
        mismatch_rate
        for backend_pair, mismatch_rate in mismatch_rates.items()
        if "beats-native-linear" in backend_pair
    ]
    custom_mismatch_rates = [
        mismatch_rate
        for backend_pair, mismatch_rate in mismatch_rates.items()
        if "beats-native-linear" not in backend_pair
    ]

    assert max(native_mismatch_rates) >= 0.2, (
        f"BEATs 原生分类头与自定义后端差异过小：{mismatch_rates}"
    )
    assert any(mismatch_rate > 0.0 for mismatch_rate in custom_mismatch_rates), (
        f"三个自定义后端之间没有产生预测差异：{mismatch_rates}"
    )


def test_classifier_backends_show_probability_profile_differences() -> None:
    """验证不同分类器后端即使类别相同也会产生不同概率分布。"""
    features, labels, probe_features = make_multimodal_embedding_case()

    results = collect_backend_results(features, labels, probe_features)
    native_result = next(
        result for result in results if result.name == "beats-native-linear"
    )
    local_density_result = next(
        result for result in results if result.name == "local-density-knn"
    )
    max_probability_delta = np.max(
        np.abs(native_result.probabilities - local_density_result.probabilities)
    )

    assert max_probability_delta > 0.05