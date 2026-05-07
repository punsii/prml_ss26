"""Percentile-model classifier for radial FFT spectra.

For each class, stores the empirical per-wavelength magnitude distribution as
100 percentile thresholds. The resulting per-class array (n_wavelengths, 100)
is the model — analogous to "weights" in a deep-learning workflow. A query
spectrum is scored by how well its values sit near the class median
(rank ≈ 50) — the MAD-from-50 metric.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def build_model(vectors: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Return {class: (n_wavelengths, 100)} percentile model built from vectors."""
    percentiles = np.arange(1, 101)
    model: dict[str, np.ndarray] = {}
    for cls in np.unique(labels):
        cls_vecs = vectors[labels == cls]
        model[cls] = np.percentile(cls_vecs, percentiles, axis=0).T
    return model


def _rank_vector(q: np.ndarray, class_model: np.ndarray) -> np.ndarray:
    """Per-bin percentile rank of q within class_model (n_wavelengths, 100). [0, 100]."""
    return (class_model < q[:, np.newaxis]).sum(axis=1)


def score_spectrum(q: np.ndarray, model: dict[str, np.ndarray]) -> dict[str, float]:
    """MAD-from-50 score per class. Higher = better fit (0 = perfect median match)."""
    return {
        cls: -float(np.mean(np.abs(_rank_vector(q, cm) - 50)))
        for cls, cm in model.items()
    }


def rank_profiles(q: np.ndarray, model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-class rank vector (n_wavelengths,) for diagnostic visualisation."""
    return {cls: _rank_vector(q, cm) for cls, cm in model.items()}


def predict(q: np.ndarray, model: dict[str, np.ndarray]) -> str:
    """Return the class with the highest MAD-from-50 score."""
    scores = score_spectrum(q, model)
    return max(scores, key=scores.__getitem__)


def mad_loo_cv(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out CV accuracy using MAD-from-50 scoring."""
    n = len(labels)
    if n < 2:
        return 0.0
    correct = sum(
        predict(
            vectors[i],
            build_model(np.delete(vectors, i, axis=0), np.delete(labels, i)),
        ) == labels[i]
        for i in range(n)
    )
    return correct / n


def save_model(
    model: dict[str, np.ndarray],
    wavelengths: np.ndarray,
    path: Path,
) -> None:
    """Serialize a percentile model to a CSV at `path`.

    Format: one row per (class, wavelength) with columns
    [class, wavelength_px, p1, p2, ..., p100].
    """
    rows = []
    p_cols = [f"p{i}" for i in range(1, 101)]
    for cls, arr in model.items():
        if arr.shape[0] != len(wavelengths):
            raise ValueError(
                f"Model for class {cls!r} has {arr.shape[0]} wavelengths but "
                f"{len(wavelengths)} were provided."
            )
        for i, wl in enumerate(wavelengths):
            row = {"class": cls, "wavelength_px": float(wl)}
            row.update(dict(zip(p_cols, arr[i].tolist())))
            rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_model(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load a percentile model from a CSV produced by `save_model`.

    Returns:
        (model, wavelengths)
        model: {class: (n_wavelengths, 100)} array of percentile thresholds.
        wavelengths: (n_wavelengths,) wavelength axis (px), sorted ascending.
    """
    df = pd.read_csv(path)
    p_cols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
    p_cols.sort(key=lambda c: int(c[1:]))
    wavelengths = np.sort(df["wavelength_px"].unique())
    model: dict[str, np.ndarray] = {}
    for cls, group in df.groupby("class"):
        ordered = group.sort_values("wavelength_px")
        model[str(cls)] = ordered[p_cols].to_numpy()
    return model, wavelengths
