"""Tests for BEATs embedding classifiers."""

from __future__ import annotations

import numpy as np
import pytest

from beats_trainer.classifiers import (
    BaseEmbeddingClassifier,
    GMMCosineKNNHybridClassifier,
    LocalDensityKNNClassifier,
    RelativeMahalanobisDistanceClassifier,
)


def make_cluster_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a tiny linearly separable embedding dataset for classifier tests."""
    rng = np.random.default_rng(42)
    normal = np.array([1.0, 0.0, 0.0, 0.0]) + rng.normal(0.0, 0.03, size=(12, 4))
    abnormal = np.array([0.0, 1.0, 0.0, 0.0]) + rng.normal(0.0, 0.03, size=(12, 4))
    features = np.vstack([normal, abnormal])
    labels = np.array(["normal"] * len(normal) + ["abnormal"] * len(abnormal))
    return features, labels


def assert_probability_output(
    classifier: BaseEmbeddingClassifier,
    features: np.ndarray,
) -> None:
    """Validate that classifier probabilities have expected shape and row sums."""
    probabilities = classifier.predict_proba(features)
    assert probabilities.shape == (features.shape[0], 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_local_density_knn_classifier_predicts_clusters() -> None:
    """Verify that LocalDensityKNNClassifier predicts simple embedding clusters."""
    features, labels = make_cluster_data()
    classifier = LocalDensityKNNClassifier(k=3, density_k=3)
    classifier.fit(features, labels)

    predictions = classifier.predict(features)

    assert np.mean(predictions == labels) >= 0.95
    assert_probability_output(classifier, features)


def test_relative_mahalanobis_classifier_predicts_clusters() -> None:
    """Verify that RelativeMahalanobisDistanceClassifier predicts simple clusters."""
    features, labels = make_cluster_data()
    classifier = RelativeMahalanobisDistanceClassifier(covariance_type="diag")
    classifier.fit(features, labels)

    predictions = classifier.predict(features)

    assert np.mean(predictions == labels) >= 0.95
    assert_probability_output(classifier, features)


def test_gmm_cosine_knn_hybrid_classifier_predicts_clusters() -> None:
    """Verify that GMMCosineKNNHybridClassifier predicts simple clusters."""
    features, labels = make_cluster_data()
    classifier = GMMCosineKNNHybridClassifier(n_components=2, knn_k=3, alpha=0.5)
    classifier.fit(features, labels)

    predictions = classifier.predict(features)

    assert np.mean(predictions == labels) >= 0.95
    assert_probability_output(classifier, features)


def test_classifier_save_and_load_round_trip(tmp_path) -> None:
    """Verify that embedding classifiers can be saved and loaded with predictions."""
    features, labels = make_cluster_data()
    classifier = LocalDensityKNNClassifier(k=3, density_k=3).fit(features, labels)
    model_path = tmp_path / "classifier.pkl"

    classifier.save(model_path)
    loaded_classifier = BaseEmbeddingClassifier.load(model_path)

    np.testing.assert_array_equal(
        loaded_classifier.predict(features),
        classifier.predict(features),
    )


def test_predict_before_fit_raises_runtime_error() -> None:
    """Verify that classifiers reject prediction before fit has been called."""
    features, _ = make_cluster_data()
    classifier = LocalDensityKNNClassifier(k=3, density_k=3)

    with pytest.raises(RuntimeError):
        classifier.predict(features)
