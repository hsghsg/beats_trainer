"""Local-density weighted KNN classifier for BEATs embeddings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

from .base import (
    BaseEmbeddingClassifier,
    as_1d_label_array,
    as_2d_float_array,
    labels_to_indices,
    row_normalize,
    validate_feature_label_shapes,
)


class LocalDensityKNNClassifier(BaseEmbeddingClassifier):
    """Classify embeddings with neighbor voting weighted by local sample density."""

    def __init__(
        self,
        k: int = 15,
        density_k: int = 20,
        metric: str = "cosine",
        epsilon: float = 1e-12,
    ) -> None:
        """Create a local-density KNN classifier with neighbor and metric settings."""
        super().__init__()
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if density_k <= 0:
            raise ValueError("density_k must be a positive integer")

        self.k = k
        self.density_k = density_k
        self.metric = metric
        self.epsilon = epsilon
        self.train_features_: np.ndarray | None = None
        self.train_label_indices_: np.ndarray | None = None
        self.local_density_: np.ndarray | None = None
        self.neighbor_model_: NearestNeighbors | None = None

    def fit(self, features: ArrayLike, labels: ArrayLike) -> "LocalDensityKNNClassifier":
        """Fit local-density statistics and the KNN index from training embeddings."""
        feature_array = as_2d_float_array(features)
        label_array = as_1d_label_array(labels)
        validate_feature_label_shapes(feature_array, label_array)

        self.classes_ = np.unique(label_array)
        self.train_features_ = feature_array
        self.train_label_indices_ = labels_to_indices(label_array, self.classes_)
        self.local_density_ = self._estimate_local_density(feature_array)
        self.neighbor_model_ = NearestNeighbors(
            n_neighbors=min(self.k, feature_array.shape[0]),
            metric=self.metric,
        )
        self.neighbor_model_.fit(feature_array)
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        """Predict class labels by selecting the largest local-density vote score."""
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, features: ArrayLike) -> np.ndarray:
        """Return normalized local-density KNN vote scores for each class."""
        scores = self._compute_vote_scores(features)
        return row_normalize(scores)

    def _estimate_local_density(self, features: np.ndarray) -> np.ndarray:
        """Estimate one density value per training sample from nearby distances."""
        neighbor_count = min(self.density_k + 1, features.shape[0])
        if neighbor_count <= 1:
            return np.ones(features.shape[0], dtype=np.float64)

        density_model = NearestNeighbors(
            n_neighbors=neighbor_count,
            metric=self.metric,
        )
        density_model.fit(features)
        distances, _ = density_model.kneighbors(features)
        neighbor_distances = distances[:, 1:]
        mean_distances = np.maximum(neighbor_distances.mean(axis=1), self.epsilon)
        return 1.0 / mean_distances

    def _compute_vote_scores(self, features: ArrayLike) -> np.ndarray:
        """Compute raw per-class vote scores for query embeddings."""
        self._check_is_fitted()
        if (
            self.neighbor_model_ is None
            or self.train_label_indices_ is None
            or self.local_density_ is None
        ):
            raise RuntimeError("Classifier internals are not fitted.")

        feature_array = as_2d_float_array(features)
        distances, indices = self.neighbor_model_.kneighbors(feature_array)
        scores = np.zeros((feature_array.shape[0], len(self.classes_)), dtype=np.float64)
        weights = self.local_density_[indices] / np.maximum(distances, self.epsilon)

        for row_index, (neighbor_indices, row_weights) in enumerate(zip(indices, weights)):
            class_indices = self.train_label_indices_[neighbor_indices]
            np.add.at(scores[row_index], class_indices, row_weights)

        return scores
