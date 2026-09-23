"""GMM and cosine-KNN hybrid classifier for BEATs embeddings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .base import (
    BaseEmbeddingClassifier,
    as_1d_label_array,
    as_2d_float_array,
    labels_to_indices,
    normalize_rows,
    row_normalize,
    stable_softmax,
    validate_feature_label_shapes,
)


class GMMCosineKNNHybridClassifier(BaseEmbeddingClassifier):
    """Combine class-wise GMM likelihoods with cosine KNN local voting."""

    def __init__(
        self,
        n_components: int = 4,
        knn_k: int = 15,
        alpha: float = 0.6,
        covariance_type: str = "diag",
        random_state: int = 42,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        epsilon: float = 1e-12,
    ) -> None:
        """Create a hybrid classifier and configure GMM, KNN, and fusion settings."""
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        if knn_k <= 0:
            raise ValueError("knn_k must be a positive integer")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        self.n_components = n_components
        self.knn_k = knn_k
        self.alpha = alpha
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.gmm_models_: list[GaussianMixture | None] = []
        self.class_means_: list[np.ndarray] = []
        self.train_features_: np.ndarray | None = None
        self.train_label_indices_: np.ndarray | None = None
        self.neighbor_model_: NearestNeighbors | None = None

    def fit(
        self,
        features: ArrayLike,
        labels: ArrayLike,
    ) -> "GMMCosineKNNHybridClassifier":
        """Fit class-wise GMMs and a cosine KNN index from embedding features."""
        feature_array = normalize_rows(as_2d_float_array(features), self.epsilon)
        label_array = as_1d_label_array(labels)
        validate_feature_label_shapes(feature_array, label_array)

        self.classes_ = np.unique(label_array)
        self.train_features_ = feature_array
        self.train_label_indices_ = labels_to_indices(label_array, self.classes_)
        self.neighbor_model_ = NearestNeighbors(
            n_neighbors=min(self.knn_k, feature_array.shape[0]),
            metric="cosine",
        )
        self.neighbor_model_.fit(feature_array)

        self.gmm_models_ = []
        self.class_means_ = []
        for class_label in self.classes_:
            class_features = feature_array[label_array == class_label]
            self.class_means_.append(class_features.mean(axis=0))
            self.gmm_models_.append(self._fit_class_gmm(class_features))

        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        """Predict labels by selecting the largest fused GMM and KNN probability."""
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, features: ArrayLike) -> np.ndarray:
        """Return fused class probabilities from GMM likelihoods and KNN votes."""
        self._check_is_fitted()
        feature_array = normalize_rows(as_2d_float_array(features), self.epsilon)
        gmm_probabilities = self._gmm_probabilities(feature_array)
        knn_probabilities = self._knn_probabilities(feature_array)
        fused_probabilities = (
            self.alpha * gmm_probabilities + (1.0 - self.alpha) * knn_probabilities
        )
        return row_normalize(fused_probabilities)

    def _fit_class_gmm(self, features: np.ndarray) -> GaussianMixture | None:
        """Fit one Gaussian mixture for a class or return None for singleton data."""
        if features.shape[0] < 2:
            return None

        component_count = min(self.n_components, features.shape[0])
        model = GaussianMixture(
            n_components=component_count,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
        )
        model.fit(features)
        return model

    def _gmm_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Compute class probabilities from GMM log-likelihood scores."""
        scores = np.column_stack(
            [
                self._score_class_density(features, model, class_mean)
                for model, class_mean in zip(self.gmm_models_, self.class_means_)
            ]
        )
        return stable_softmax(scores)

    def _score_class_density(
        self,
        features: np.ndarray,
        model: GaussianMixture | None,
        class_mean: np.ndarray,
    ) -> np.ndarray:
        """Score one class with a fitted GMM or a centroid fallback."""
        if model is not None:
            return model.score_samples(features)

        centered = features - class_mean
        return -np.sum(centered * centered, axis=1) / max(self.reg_covar, self.epsilon)

    def _knn_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Compute class probabilities from inverse-distance cosine KNN voting."""
        if self.neighbor_model_ is None or self.train_label_indices_ is None:
            raise RuntimeError("Classifier internals are not fitted.")

        distances, indices = self.neighbor_model_.kneighbors(features)
        scores = np.zeros((features.shape[0], len(self.classes_)), dtype=np.float64)
        weights = 1.0 / np.maximum(distances, self.epsilon)

        for row_index, (neighbor_indices, row_weights) in enumerate(zip(indices, weights)):
            class_indices = self.train_label_indices_[neighbor_indices]
            np.add.at(scores[row_index], class_indices, row_weights)

        return row_normalize(scores)
