"""Shared interfaces and helpers for BEATs embedding classifiers."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class BaseEmbeddingClassifier(ABC):
    """Define the common contract for classifiers trained on BEATs embeddings."""

    classes_: np.ndarray | None

    def __init__(self) -> None:
        """Initialize the classifier base state before a concrete estimator is fitted."""
        self.classes_ = None

    @abstractmethod
    def fit(self, features: ArrayLike, labels: ArrayLike) -> "BaseEmbeddingClassifier":
        """Fit the classifier from a two-dimensional embedding matrix and labels."""

    @abstractmethod
    def predict(self, features: ArrayLike) -> np.ndarray:
        """Predict class labels for a two-dimensional embedding matrix."""

    @abstractmethod
    def predict_proba(self, features: ArrayLike) -> np.ndarray:
        """Return per-class probabilities or normalized scores for each embedding."""

    def save(self, model_path: str | Path) -> None:
        """Serialize the fitted classifier to disk with pickle for later inference."""
        target_path = Path(model_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as output_file:
            pickle.dump(self, output_file)

    @classmethod
    def load(cls, model_path: str | Path) -> "BaseEmbeddingClassifier":
        """Load a serialized embedding classifier from disk and return the estimator."""
        with Path(model_path).open("rb") as input_file:
            model = pickle.load(input_file)

        if not isinstance(model, BaseEmbeddingClassifier):
            raise TypeError(
                f"Loaded object is not a BaseEmbeddingClassifier: {type(model)!r}"
            )

        return model

    def _check_is_fitted(self) -> None:
        """Validate that the classifier has learned class metadata before inference."""
        if self.classes_ is None:
            raise RuntimeError("Classifier has not been fitted yet.")


def as_2d_float_array(features: ArrayLike) -> np.ndarray:
    """Convert input features to a finite two-dimensional float array."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"features must be a 2D array, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("features must contain at least one sample")
    if not np.isfinite(array).all():
        raise ValueError("features must not contain NaN or infinity values")
    return array


def as_1d_label_array(labels: ArrayLike) -> np.ndarray:
    """Convert input labels to a non-empty one-dimensional numpy array."""
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"labels must be a 1D array, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("labels must contain at least one sample")
    return array


def validate_feature_label_shapes(features: np.ndarray, labels: np.ndarray) -> None:
    """Ensure feature rows and label count describe the same number of samples."""
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "features and labels must contain the same number of samples: "
            f"{features.shape[0]} != {labels.shape[0]}"
        )


def normalize_rows(features: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """L2-normalize each feature row while keeping zero vectors numerically stable."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, epsilon)


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary scores to row-wise probabilities with a stable softmax."""
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)


def row_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize non-negative row scores and fall back to uniform probabilities."""
    row_sums = scores.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        scores,
        row_sums,
        out=np.zeros_like(scores, dtype=np.float64),
        where=row_sums > 0,
    )

    empty_rows = np.where(row_sums.ravel() <= 0)[0]
    if empty_rows.size > 0:
        probabilities[empty_rows] = 1.0 / scores.shape[1]

    return probabilities


def labels_to_indices(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map labels to integer class indices according to an existing class array."""
    class_to_index: dict[Any, int] = {
        class_label: index for index, class_label in enumerate(classes)
    }
    return np.array([class_to_index[label] for label in labels], dtype=np.int64)
