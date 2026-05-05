"""Percentile atlas classifier for radial FFT spectra.

For each class, stores the empirical per-bin magnitude distribution as 100
percentile thresholds.  A query spectrum is scored by how well its values sit
near the class median (rank ≈ 50) — the MAD-from-50 metric.
"""

import numpy as np


def build_atlas(vectors: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Return {class: (n_bins, 100)} percentile atlas built from vectors."""
    percentiles = np.arange(1, 101)
    atlas = {}
    for cls in np.unique(labels):
        cls_vecs = vectors[labels == cls]
        atlas[cls] = np.percentile(cls_vecs, percentiles, axis=0).T  # (n_bins, 100)
    return atlas


def _rank_vector(q: np.ndarray, class_atlas: np.ndarray) -> np.ndarray:
    """Per-bin percentile rank of q within class_atlas (n_bins, 100). Result in [0, 100]."""
    return (class_atlas < q[:, np.newaxis]).sum(axis=1)


def score_spectrum(q: np.ndarray, atlas: dict[str, np.ndarray]) -> dict[str, float]:
    """MAD-from-50 score per class. Higher = better fit (0 = perfect median match)."""
    return {
        cls: -float(np.mean(np.abs(_rank_vector(q, ca) - 50)))
        for cls, ca in atlas.items()
    }


def rank_profiles(q: np.ndarray, atlas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-class rank vector (n_bins,) for diagnostic visualisation."""
    return {cls: _rank_vector(q, ca) for cls, ca in atlas.items()}


def predict(q: np.ndarray, atlas: dict[str, np.ndarray]) -> str:
    """Return the class with the highest MAD-from-50 score."""
    scores = score_spectrum(q, atlas)
    return max(scores, key=scores.__getitem__)


def mad_loo_cv(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out CV accuracy using MAD-from-50 scoring."""
    n = len(labels)
    if n < 2:
        return 0.0
    correct = sum(
        predict(
            vectors[i],
            build_atlas(np.delete(vectors, i, axis=0), np.delete(labels, i)),
        ) == labels[i]
        for i in range(n)
    )
    return correct / n
