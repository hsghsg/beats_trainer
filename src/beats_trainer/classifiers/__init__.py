"""Embedding classifiers for BEATs feature vectors."""

from .base import BaseEmbeddingClassifier
from .gmm_cosine_knn import GMMCosineKNNHybridClassifier
from .local_density_knn import LocalDensityKNNClassifier
from .relative_mahalanobis import RelativeMahalanobisDistanceClassifier

__all__ = [
    "BaseEmbeddingClassifier",
    "GMMCosineKNNHybridClassifier",
    "LocalDensityKNNClassifier",
    "RelativeMahalanobisDistanceClassifier",
]
