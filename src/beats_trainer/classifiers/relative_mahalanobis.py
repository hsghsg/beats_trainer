"""Relative Mahalanobis distance classifier for BEATs embeddings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import (
    BaseEmbeddingClassifier,
    as_1d_label_array,
    as_2d_float_array,
    stable_softmax,
    validate_feature_label_shapes,
)


class RelativeMahalanobisDistanceClassifier(BaseEmbeddingClassifier):
    """Classify embeddings by relative class-wise Mahalanobis distance."""

    def __init__(
        self,
        covariance_type: str = "diag",
        regularization: float = 1e-4,
    ) -> None:
        """Create a Mahalanobis classifier with diagonal or full covariance."""
        super().__init__()
        if covariance_type not in {"diag", "full"}:
            raise ValueError("covariance_type must be 'diag' or 'full'")
        if regularization <= 0:
            raise ValueError("regularization must be positive")

        self.covariance_type = covariance_type
        self.regularization = regularization
        self.class_means_: list[np.ndarray] = []
        self.class_inverse_covariances_: list[np.ndarray] = []
        self.global_mean_: np.ndarray | None = None
        self.global_inverse_covariance_: np.ndarray | None = None

    def fit(
        self,
        features: ArrayLike,
        labels: ArrayLike,
    ) -> "RelativeMahalanobisDistanceClassifier":
        """Estimate class and global Gaussian statistics from embedding features."""
        feature_array = as_2d_float_array(features)
        label_array = as_1d_label_array(labels)
        validate_feature_label_shapes(feature_array, label_array)

        self.classes_ = np.unique(label_array)
        self.class_means_ = []
        self.class_inverse_covariances_ = []

        for class_label in self.classes_:
            class_features = feature_array[label_array == class_label]
            self.class_means_.append(class_features.mean(axis=0))
            self.class_inverse_covariances_.append(
                self._estimate_inverse_covariance(class_features)
            )

        self.global_mean_ = feature_array.mean(axis=0)
        self.global_inverse_covariance_ = self._estimate_inverse_covariance(feature_array)
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        """Predict labels by selecting the class with the largest relative score."""
        scores = self.decision_function(features)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, features: ArrayLike) -> np.ndarray:
        """Convert relative Mahalanobis scores to class probabilities."""
        return stable_softmax(self.decision_function(features))

    def decision_function(self, features: ArrayLike) -> np.ndarray:
        """Return larger-is-better scores from global minus class distance."""
        self._check_is_fitted()
        if self.global_mean_ is None or self.global_inverse_covariance_ is None:
            raise RuntimeError("Classifier internals are not fitted.")

        feature_array = as_2d_float_array(features)
        global_distances = self._mahalanobis_distance(
            feature_array,
            self.global_mean_,
            self.global_inverse_covariance_,
        )
        class_distances = []
        for mean, inverse_covariance in zip(
            self.class_means_,
            self.class_inverse_covariances_,
        ):
            class_distances.append(
                self._mahalanobis_distance(feature_array, mean, inverse_covariance)
            )

        class_distance_array = np.column_stack(class_distances)
        return global_distances[:, None] - class_distance_array

    def _estimate_inverse_covariance(self, features: np.ndarray) -> np.ndarray:
        """Estimate a regularized inverse covariance matrix or diagonal vector."""
        if self.covariance_type == "diag":
            variance = np.var(features, axis=0) + self.regularization
            return 1.0 / variance

        if features.shape[0] <= 1:
            covariance = np.eye(features.shape[1]) * self.regularization
        else:
            covariance = np.cov(features, rowvar=False)
            covariance = np.atleast_2d(covariance)
            covariance += np.eye(features.shape[1]) * self.regularization

        return np.linalg.pinv(covariance)

    def _mahalanobis_distance(
        self,
        features: np.ndarray,
        mean: np.ndarray,
        inverse_covariance: np.ndarray,
    ) -> np.ndarray:
        """Compute squared Mahalanobis distances for every feature row."""
        centered = features - mean
        if inverse_covariance.ndim == 1:
            return np.sum(centered * centered * inverse_covariance, axis=1)
        return np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered)
