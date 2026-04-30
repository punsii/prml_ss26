"""Streamlit app: compare brightness normalization methods in patch mode (grayscale).

Each image is loaded as grayscale and split into a 9x9 grid; the 9 tiles at
row/col indices (1, 4, 7) are processed independently. Outer tiles are
unused so kernels on the selected tiles never need padding from outside
the image.
"""

import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clahe import apply_clahe
from class_stats import (CLASS_COLORS, compute_class_stats,
                         compute_full_image_vectors, load_labels, logreg_cv,
                         mahalanobis_filter)
from homomorphic import homomorphic_filter
from radial import (GRID_SIZE, RADIAL_DC_TRIM, TILE_INDICES, extract_patches,
                    radial_profile)
from retinex import multi_scale_retinex

IMAGE_DIR = Path("data/BMW_25/Rohdaten/Erste Bearbeitungsstufe 10-17ym")

CENTER_IDX = (len(TILE_INDICES) ** 2) // 2  # row-major index of (4,4) tile

METHODS = {
    "CLAHE": (
        apply_clahe,
        "Tile-based histogram equalization with contrast clipping. Cheap, no spectral assumptions.",
    ),
    "Retinex": (
        multi_scale_retinex,
        "Log-domain illumination subtraction at multiple Gaussian scales (15/80/250).",
    ),
    "Homomorphic": (
        homomorphic_filter,
        "Gaussian high-pass in the log-frequency domain. Suppresses slow illumination, keeps texture.",
    ),
}


def run_on_patches(func, image: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Time full pipeline: split + process all 9 tiles."""
    t0 = time.perf_counter()
    results = [func(p) for p in extract_patches(image)]
    return results, time.perf_counter() - t0


def assemble_full(image: np.ndarray, processed: list[np.ndarray]) -> np.ndarray:
    """Insert the 9 processed tiles into a copy of the image at their grid positions."""
    out = image.copy()
    h, w = image.shape[:2]
    th, tw = h // GRID_SIZE, w // GRID_SIZE
    for k, (i, j) in enumerate((i, j) for i in TILE_INDICES for j in TILE_INDICES):
        out[i * th : (i + 1) * th, j * tw : (j + 1) * tw] = processed[k]
    return out


def draw_tile_outlines(image: np.ndarray) -> np.ndarray:
    """Return RGB image with red rectangles around selected tiles (input is grayscale)."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    th, tw = h // GRID_SIZE, w // GRID_SIZE
    for i in TILE_INDICES:
        for j in TILE_INDICES:
            cv2.rectangle(
                out,
                (j * tw, i * th),
                ((j + 1) * tw, (i + 1) * th),
                color=(0, 0, 255),
                thickness=20,
            )
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def render_tile_grid(images: list[np.ndarray], label: str) -> None:
    """Render a 3x3 grid of tile images under a header (~70% of container width)."""
    st.subheader(label)
    n = len(TILE_INDICES)
    for row in range(n):
        _, mid, _ = st.columns([1, 5, 1])
        with mid:
            cols = st.columns(n)
            for col in range(n):
                cols[col].image(images[row * n + col], use_container_width=True)


def render_chart_grid(profiles: list[np.ndarray], label: str) -> None:
    """3x3 grid of radial-spectrum line charts (~70% of container width, log y)."""
    st.subheader(label)
    n = len(TILE_INDICES)
    for row in range(n):
        _, mid, _ = st.columns([1, 5, 1])
        with mid:
            cols = st.columns(n)
            for col in range(n):
                cols[col].line_chart(np.log1p(profiles[row * n + col][RADIAL_DC_TRIM:]))


def render_comparison(image: np.ndarray) -> None:
    """Per method: short description, full-image stitched view, then center-tile detail."""
    original_center = extract_patches(image)[CENTER_IDX]

    for name, (func, description) in METHODS.items():
        st.markdown(f"### {name}\n_{description}_")
        results, elapsed = run_on_patches(func, image)
        full = assemble_full(image, results)

        _, mid, _ = st.columns([1, 2, 1])
        mid.image(
            full,
            caption=f"Full image with {name} applied to the 9 selected tiles",
            use_container_width=True,
        )

        cols = st.columns(4)
        cols[0].image(
            original_center, caption="Original (center)", use_container_width=True
        )
        cols[1].image(
            results[CENTER_IDX], caption=f"{name} (center)", use_container_width=True
        )
        with cols[2]:
            st.line_chart(
                np.log1p(radial_profile(results[CENTER_IDX])[RADIAL_DC_TRIM:])
            )
            st.caption(f"Log radial spectrum ({name})")
        cols[3].metric(f"{name} — 9 tiles", f"{elapsed:.3f}s")


def render_homomorphic_detail(image: np.ndarray) -> None:
    """Full original with tile outlines, then stacked grids: originals, processed, radial spectra."""
    st.markdown(
        "_Per-tile homomorphic results with the full-image context and radial spectra of all tiles._"
    )

    _, mid, _ = st.columns([1, 2, 1])
    mid.image(
        draw_tile_outlines(image),
        caption="Full original (red = selected tiles)",
        use_container_width=True,
    )

    patches = extract_patches(image)
    results, elapsed = run_on_patches(homomorphic_filter, image)
    st.metric("Homomorphic — time (9 tiles)", f"{elapsed:.3f}s")

    render_tile_grid(patches, "Original")
    render_tile_grid(results, "Processed")
    render_chart_grid(
        [radial_profile(r) for r in results], "Radial spectrum (processed)"
    )


LABOR_DIR = Path("data/Datensatz_Labor")
LABOR_CSV = Path("data/Labels_Datensatz_Labr.csv")

METHOD_FNS = {
    "raw": None,
    "clahe": apply_clahe,
    "retinex": multi_scale_retinex,
    "homomorphic": homomorphic_filter,
}


@st.cache_data(show_spinner="Computing spectra…")
def _cached_vectors(method: str, center_percentage: float) -> dict | None:
    """Cache-keyed on (method, center_percentage). Returns raw vector data."""
    if not LABOR_DIR.exists() or not LABOR_CSV.exists():
        return None
    rows = load_labels(LABOR_CSV)
    fn = METHOD_FNS.get(method)
    vectors, labels, filenames = compute_full_image_vectors(
        rows, LABOR_DIR, fn, center_percentage
    )
    if len(vectors) == 0:
        return None
    return {"vectors": vectors, "labels": labels, "filenames": filenames}


def render_class_spectra_tab() -> None:
    st.header("Class spectra (Datensatz_Labor)")

    if not LABOR_DIR.exists():
        st.warning(f"Data dir not found: `{LABOR_DIR}`. Run from repo root.")
        return
    if not LABOR_CSV.exists():
        st.warning(f"Labels CSV not found: `{LABOR_CSV}`.")
        return

    method = st.selectbox(
        "Method",
        list(METHOD_FNS.keys()),
        index=0,
        help="Preprocessing applied to each image before computing radial spectrum. 'raw' = no preprocessing.",
    )
    center_percentage = st.slider(
        "Center percentage",
        0.3,
        1.0,
        0.5,
        0.05,
        help="Fraction of image center used for radial spectrum (width and height).",
    )
    drop_percentage = st.slider(
        "Drop percentage",
        0.1,
        1.0,
        0.1,
        0.01,
        help="Fraction of each class to drop as outliers (Mahalanobis distance).",
    )

    if not st.button("Compute"):
        st.info("Press 'Compute' to run the analysis.")
        return

    progress = st.progress(0, text="Loading images…")
    cached = _cached_vectors(method, center_percentage)
    progress.progress(50, text="Filtering outliers…")

    if cached is None:
        st.error("No images found or data unavailable.")
        progress.empty()
        return

    vectors = cached["vectors"]
    labels = cached["labels"]
    filenames = cached["filenames"]

    kept_mask, distances, dropped_per_class = mahalanobis_filter(
        vectors, labels, drop_percentage
    )

    vectors_f = vectors[kept_mask]
    labels_f = labels[kept_mask]
    progress.progress(75, text="Computing statistics…")

    class_names = sorted(np.unique(labels).tolist())
    class_stats = compute_class_stats(vectors_f, labels_f)
    bin_axis = np.arange(RADIAL_DC_TRIM, vectors_f.shape[1])

    # --- Plot 1: Per-class radial spectra (Plotly line chart, log y) ---
    fig_spec = go.Figure()
    for cls in class_names:
        color = CLASS_COLORS.get(cls, "gray")
        if cls not in class_stats:
            continue
        m = class_stats[cls]["mean"][RADIAL_DC_TRIM:]
        s = class_stats[cls]["std"][RADIAL_DC_TRIM:]
        x = bin_axis.tolist()
        fig_spec.add_trace(
            go.Scatter(
                x=x,
                y=m.tolist(),
                mode="lines",
                name=cls,
                line=dict(color=color),
            )
        )
        fig_spec.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=np.maximum(m + s, 1e-6).tolist() + np.maximum(m - s, 1e-6).tolist(),
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig_spec.update_yaxes(type="log")
    fig_spec.update_layout(
        title=f"Per-class radial spectra — {method}",
        xaxis_title="Radial bin",
        yaxis_title="Mean FFT magnitude (log)",
        height=400,
    )
    st.plotly_chart(fig_spec, use_container_width=True)

    # --- Plot 2: PCA 3D scatter (Plotly, interactive) ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(vectors_f)
    pca = PCA(n_components=3)
    Xp = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_

    fig_pca = go.Figure()
    for cls in class_names:
        mask = labels_f == cls
        if not mask.any():
            continue
        color = CLASS_COLORS.get(cls, "gray")
        fig_pca.add_trace(
            go.Scatter3d(
                x=Xp[mask, 0].tolist(),
                y=Xp[mask, 1].tolist(),
                z=Xp[mask, 2].tolist(),
                mode="markers",
                name=cls,
                marker=dict(size=3, color=color, opacity=0.6),
            )
        )
    fig_pca.update_layout(
        title=f"PCA 3D — {method}  (expl. var: {sum(evr):.1%})",
        scene=dict(
            xaxis_title=f"PC1 ({evr[0]:.1%})",
            yaxis_title=f"PC2 ({evr[1]:.1%})",
            zaxis_title=f"PC3 ({evr[2]:.1%})",
        ),
        height=600,
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # --- Logreg on filtered data ---
    acc, std = logreg_cv(vectors_f, labels_f)
    st.subheader("Logistic regression CV accuracy (filtered)")
    st.caption(
        "Multinomial logistic regression on filtered radial profile vectors. "
        "5-fold stratified CV. Random baseline = 1/4 = 0.25."
    )
    df_lr = pd.DataFrame(
        [
            {
                "Method": method,
                "Center percentage": center_percentage,
                "Drop percentage": drop_percentage,
                "Kept / Total": f"{kept_mask.sum()} / {len(kept_mask)}",
                "Mean accuracy": f"{acc:.4f}",
                "Std": f"{std:.4f}",
            }
        ]
    )
    st.dataframe(df_lr, use_container_width=True)

    progress.progress(100, text="Done.")

    # --- Dropped thumbnails ---
    n_dropped_total = (~kept_mask).sum()
    if n_dropped_total > 0:
        st.subheader(f"Dropped images ({n_dropped_total} total)")
        for cls in class_names:
            dropped_indices = dropped_per_class.get(cls, [])
            if not dropped_indices:
                continue
            with st.expander(f"{cls} — {len(dropped_indices)} dropped"):
                show_indices = dropped_indices[:24]
                cols = st.columns(min(6, len(show_indices)))
                for k, idx in enumerate(show_indices):
                    fname = filenames[idx]
                    img_path = LABOR_DIR / fname
                    col = cols[k % 6]
                    if img_path.exists():
                        col.image(
                            str(img_path), caption=f"d={distances[idx]:.2f}", width=100
                        )
                    else:
                        col.text(f"{fname}\nd={distances[idx]:.2f}")


def main():
    st.set_page_config(layout="wide", page_title="Reflection Removal Comparison")
    st.title("Brightness Normalization Comparison (patch mode, grayscale)")

    if not IMAGE_DIR.exists():
        st.error(f"Image directory not found: `{IMAGE_DIR}`")
        st.info("Run this app from the repo root directory.")
        return

    files = sorted(
        f.name for f in IMAGE_DIR.iterdir() if f.suffix.upper() in (".JPG", ".PNG")
    )
    if not files:
        st.error("No images found.")
        return

    default_idx = files.index("_DSC1090.JPG") if "_DSC1090.JPG" in files else 0
    selected = st.selectbox("Select image", files, index=default_idx)
    image = cv2.imread(str(IMAGE_DIR / selected), cv2.IMREAD_GRAYSCALE)
    if image is None:
        st.error(f"Cannot read: {selected}")
        return

    tab_cmp, tab_homo, tab_cls = st.tabs(
        ["Method comparison", "Homomorphic detail", "Class spectra (Datensatz_Labor)"]
    )
    with tab_cmp:
        render_comparison(image)
    with tab_homo:
        render_homomorphic_detail(image)
    with tab_cls:
        render_class_spectra_tab()


if __name__ == "__main__":
    main()
