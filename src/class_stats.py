"""Pure library for class-level spectral statistics and outlier filtering."""

import csv
import re
from pathlib import Path

import cv2
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from radial import (RADIAL_DC_TRIM, extract_patches,
                    radial_profile_center_fraction)

CLASS_COLORS = {
    "1-3um": "red",
    "3-5um": "green",
    "5-7um": "blue",
    "7+um": "gold",
}

METHOD_FNS: dict  # populated by app.py after importing clahe/retinex/homomorphic


def _normalize_filename(name: str) -> str:
    return re.sub(r"_(\d+)\.JPG$", r"_D\1.JPG", name)


def load_labels(csv_path: Path) -> list[tuple[str, str]]:
    """Read CSV (delimiter ';'), columns Filename and Label.

    Skip rows where Label is empty or 'Label'. Apply _normalize_filename to filenames.
    Return list of (normalized_filename, label).
    """
    rows: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            label = r.get("Label", "").strip()
            filename = r.get("Filename", "").strip()
            if label in ("", "Label") or not filename:
                continue
            rows.append((_normalize_filename(filename), label))
    return rows


def load_labels_from_dirs(root_dir: Path) -> list[tuple[str, str]]:
    """Derive class labels from subfolder names under root_dir.

    Each immediate subdirectory of root_dir is treated as a class.
    Returns list of (relative_path, label) where relative_path is
    "<subfolder_name>/<filename>" — joinable with root_dir to get the
    full image path.

    Supported extensions: .jpg, .jpeg, .png, .bmp (case-insensitive).
    """

    def _safe(s: str) -> str:
        return s.encode("utf-8", "surrogateescape").decode("utf-8", "replace")

    rows: list[tuple[str, str]] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label = _safe(subdir.name)
        for img_file in sorted(subdir.iterdir()):
            if img_file.suffix.lower() in exts:
                rows.append((f"{subdir.name}/{img_file.name}", label))
    return rows


def compute_full_image_vectors(
    rows: list[tuple[str, str]],
    data_dir: Path,
    method_fn,
    center_percentage: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute per-image radial profile vectors from full images.

    Args:
        rows: list of (filename, label) from load_labels.
        data_dir: directory containing the images.
        method_fn: callable applied to full grayscale image before profiling, or None.
        center_percentage: fraction of image center to use for radial profile.

    Returns:
        (vectors_2d_array, labels_array, filenames_list) — all length N.
        vectors_2d_array has shape (N, min_profile_len).
    """
    profiles: list[np.ndarray] = []
    labels: list[str] = []
    filenames: list[str] = []

    for fname, label in rows:
        path = data_dir / fname
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if method_fn is not None:
            img = method_fn(img)
        profile = radial_profile_center_fraction(img, center_percentage)
        profiles.append(profile)
        labels.append(label)
        filenames.append(fname)

    if not profiles:
        return np.empty((0, 0)), np.empty(0, dtype=str), []

    min_len = min(len(p) for p in profiles)
    vectors = np.array([p[:min_len] for p in profiles])
    return vectors, np.array(labels), filenames


def compute_class_stats(vectors: np.ndarray, labels: np.ndarray) -> dict:
    """Compute per-class mean and std.

    Returns:
        {class_name: {"mean": ndarray, "std": ndarray}}
    """
    stats: dict = {}
    for cls in np.unique(labels):
        mask = labels == cls
        mat = vectors[mask]
        stats[cls] = {"mean": mat.mean(axis=0), "std": mat.std(axis=0)}
    return stats


def mahalanobis_filter(
    vectors: np.ndarray,
    labels: np.ndarray,
    drop_percentage: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Filter outliers per class using Mahalanobis distance.

    Args:
        vectors: shape (N, D).
        labels: shape (N,).
        drop_percentage: fraction of each class to drop (top outliers by distance).

    Returns:
        kept_mask: bool array (N,), True = kept.
        distances: float array (N,) Mahalanobis distance per sample.
        dropped_per_class: {class_name: list of original indices dropped}.
    """
    n = len(labels)
    kept_mask = np.ones(n, dtype=bool)
    distances = np.zeros(n, dtype=float)
    dropped_per_class: dict = {}

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        cls_vecs = vectors[cls_idx]

        if len(cls_idx) < 2:
            dropped_per_class[cls] = []
            continue

        lw = LedoitWolf(store_precision=True)
        lw.fit(cls_vecs)
        mu = lw.location_
        prec = lw.precision_

        diffs = cls_vecs - mu
        # vectorised Mahalanobis using precomputed precision matrix
        d = np.sqrt(np.einsum("ij,jk,ik->i", diffs, prec, diffs))
        distances[cls_idx] = d

        n_drop = int(np.floor(len(cls_idx) * drop_percentage))
        if n_drop > 0:
            sorted_order = np.argsort(d)[::-1]
            drop_local = sorted_order[:n_drop]
            drop_global = cls_idx[drop_local]
            kept_mask[drop_global] = False
            dropped_per_class[cls] = drop_global.tolist()
        else:
            dropped_per_class[cls] = []

    return kept_mask, distances, dropped_per_class


def logreg_cv(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic regression cross-validated accuracy.

    StandardScaler → LogisticRegression(max_iter=1000) → StratifiedKFold CV.
    n_splits = min(5, min_class_count); returns (0.0, 0.0) if n_splits < 2.

    Returns:
        (mean_accuracy, std_accuracy)
    """
    from collections import Counter

    min_class_count = min(Counter(y).values())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        return 0.0, 0.0

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())
