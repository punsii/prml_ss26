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
                         compute_full_image_vectors, load_labels,
                         load_labels_from_dirs, logreg_cv, mahalanobis_filter)
from homomorphic import homomorphic_filter
from radial import (GRID_SIZE, RADIAL_DC_TRIM, TILE_INDICES, _radial_from_mag,
                    center_crop, extract_patches, radial_profile)
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
BMW_DIR = Path("data/BMW_25/Rohdaten")

DATASETS = {
    "Datensatz Labor": {
        "type": "csv",
        "image_dir": LABOR_DIR,
        "csv_path": LABOR_CSV,
    },
    "BMW 25": {
        "type": "dirs",
        "image_dir": BMW_DIR,
    },
}

BMW_CLASS_COLORS = {
    "Fr+\ufffdszustand 17-25ym": "red",
    "Erste Bearbeitungsstufe 10-17ym": "orange",
    "Zweite Bearbeitungsstufe 3-10ym": "green",
    "Finaler Zustand kleiner 3ym": "blue",
}

METHOD_FNS = {
    "raw": None,
    "clahe": apply_clahe,
    "retinex": multi_scale_retinex,
    "homomorphic": homomorphic_filter,
}


@st.cache_data(show_spinner="Computing spectra…")
def _cached_vectors(
    dataset_name: str, method: str, center_percentage: float
) -> dict | None:
    """Cache-keyed on (dataset_name, method, center_percentage). Returns raw vector data."""
    cfg = DATASETS[dataset_name]
    image_dir = cfg["image_dir"]
    if not image_dir.exists():
        return None

    if cfg["type"] == "csv":
        csv_path = cfg["csv_path"]
        if not csv_path.exists():
            return None
        rows = load_labels(csv_path)
    else:
        rows = load_labels_from_dirs(image_dir)

    fn = METHOD_FNS.get(method)
    vectors, labels, filenames = compute_full_image_vectors(
        rows, image_dir, fn, center_percentage
    )
    if len(vectors) == 0:
        return None
    return {"vectors": vectors, "labels": labels, "filenames": filenames}


def _render_fft_explorer(
    dataset_name: str, method: str, center_percentage: float
) -> None:
    """Show 2D FFT magnitude + radial profile for a single selected image."""
    cfg = DATASETS[dataset_name]
    image_dir = cfg["image_dir"]
    if not image_dir.exists():
        st.warning("Dataset directory not found.")
        return

    if cfg["type"] == "csv":
        rows = load_labels(cfg["csv_path"]) if cfg["csv_path"].exists() else []
        img_paths = [
            image_dir / fname for fname, _ in rows if (image_dir / fname).exists()
        ]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_paths = sorted(
            p
            for subdir in sorted(image_dir.iterdir())
            if subdir.is_dir()
            for p in sorted(subdir.iterdir())
            if p.suffix.lower() in exts
        )

    if not img_paths:
        st.warning("No images found for FFT explorer.")
        return

    def _safe(s: str) -> str:
        """Re-encode surrogate-escaped bytes (from broken locale) as UTF-8."""
        return s.encode("utf-8", "surrogateescape").decode("utf-8", "replace")

    if cfg["type"] == "dirs":
        labels_display = [_safe(f"[{p.parent.name}] {p.name}") for p in img_paths]
    else:
        labels_display = [_safe(str(p.relative_to(image_dir))) for p in img_paths]
    sel_idx = st.selectbox(
        "Image",
        range(len(labels_display)),
        format_func=lambda i: labels_display[i],
    )
    selected_path = img_paths[sel_idx]

    raw_img = cv2.imread(str(selected_path), cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        st.error(f"Cannot read: {selected_path}")
        return

    fn = METHOD_FNS.get(method)
    img = fn(raw_img) if fn is not None else raw_img

    cropped = center_crop(img, center_percentage)
    h, w = cropped.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(cropped.astype(np.float32) * win)))

    col_orig, col_a, col_b = st.columns(3)
    with col_orig:
        st.image(raw_img, caption="Original", use_container_width=True)
    with col_a:
        fig_2d = go.Figure(
            go.Heatmap(z=mag.tolist(), colorscale="Viridis", showscale=False)
        )
        fig_2d.update_layout(
            title="2D FFT magnitude",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, scaleanchor="x"),
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_2d, use_container_width=True)
    with col_b:
        profile = _radial_from_mag(mag)
        x_bins = list(range(len(profile)))
        fig_rad = go.Figure(
            go.Scatter(x=x_bins, y=profile.tolist(), mode="lines", name="magnitude")
        )
        fig_rad.update_yaxes(type="log")
        fig_rad.update_layout(
            title="Radial profile (log y)",
            xaxis_title="Radial bin",
            yaxis_title="Magnitude",
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_rad, use_container_width=True)


def render_class_spectra_tab() -> None:
    st.header("Class spectra")

    # --- Dataset selector ---
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), index=0)
    cfg = DATASETS[dataset_name]
    image_dir: Path = cfg["image_dir"]
    colors = CLASS_COLORS if dataset_name == "Datensatz Labor" else BMW_CLASS_COLORS

    if not image_dir.exists():
        st.warning(f"Dataset directory not found: `{image_dir}`. Run from repo root.")
        return
    if cfg["type"] == "csv" and not cfg["csv_path"].exists():
        st.warning(f"Labels CSV not found: `{cfg['csv_path']}`.")
        return

    # --- Method + spectrum controls ---
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
        help="Fraction of image center (width and height) used for radial spectrum.",
    )

    # --- 2D FFT explorer ---
    with st.expander("2D power spectrum explorer", expanded=False):
        _render_fft_explorer(dataset_name, method, center_percentage)

    # --- Frequency band sliders ---
    st.markdown("**Frequency band for analysis**")
    freq_col1, freq_col2 = st.columns(2)
    with freq_col1:
        freq_min = st.number_input(
            "Freq min bin",
            min_value=0,
            max_value=500,
            value=RADIAL_DC_TRIM,
            step=1,
            help="Lowest radial bin included. Use the FFT explorer to identify the grinding line band.",
        )
    with freq_col2:
        freq_max = st.number_input(
            "Freq max bin",
            min_value=1,
            max_value=2000,
            value=200,
            step=1,
            help="Highest radial bin included (exclusive). Large value = include all.",
        )

    drop_percentage = st.slider(
        "Drop percentage",
        0.0,
        1.0,
        0.1,
        0.01,
        help="Fraction of each class to drop as Mahalanobis outliers.",
    )

    if not st.button("Compute"):
        st.info("Press 'Compute' to run the analysis.")
        return

    progress = st.progress(0, text="Loading images…")
    cached = _cached_vectors(dataset_name, method, center_percentage)
    progress.progress(40, text="Filtering outliers…")

    if cached is None:
        st.error("No images found or data unavailable.")
        progress.empty()
        return

    vectors = cached["vectors"]
    labels = cached["labels"]
    filenames = cached["filenames"]

    # Band-limit
    actual_max = vectors.shape[1]
    fmin = int(freq_min)
    fmax = min(int(freq_max), actual_max)
    if fmin >= fmax:
        st.error(
            f"freq_min ({fmin}) must be less than freq_max ({fmax}). Vector length: {actual_max}."
        )
        progress.empty()
        return

    vectors_b = vectors[:, fmin:fmax]
    bin_axis = list(range(fmin, fmax))

    kept_mask, distances, dropped_per_class = mahalanobis_filter(
        vectors_b, labels, drop_percentage
    )

    vectors_f = vectors_b[kept_mask]
    labels_f = labels[kept_mask]
    progress.progress(70, text="Computing statistics…")

    class_names = sorted(np.unique(labels).tolist())
    class_stats_data = compute_class_stats(vectors_f, labels_f)

    # --- Plot 1: Per-class radial spectra ---
    fig_spec = go.Figure()
    for cls in class_names:
        color = colors.get(cls, "gray")
        if cls not in class_stats_data:
            continue
        m = class_stats_data[cls]["mean"]
        s = class_stats_data[cls]["std"]
        fig_spec.add_trace(
            go.Scatter(
                x=bin_axis,
                y=m.tolist(),
                mode="lines",
                name=cls,
                line=dict(color=color),
            )
        )
        fig_spec.add_trace(
            go.Scatter(
                x=bin_axis + bin_axis[::-1],
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
        title=f"Per-class radial spectra — {dataset_name} — {method}  [bins {fmin}:{fmax}]",
        xaxis_title="Radial bin",
        yaxis_title="Mean FFT magnitude (log)",
        height=400,
    )
    st.plotly_chart(fig_spec, use_container_width=True)

    # --- Plot 2: PCA 3D scatter ---
    if vectors_f.shape[0] >= 3 and vectors_f.shape[1] >= 3:
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
            color = colors.get(cls, "gray")
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
            title=f"PCA 3D — {dataset_name} — {method}  (expl. var: {sum(evr):.1%})",
            scene=dict(
                xaxis_title=f"PC1 ({evr[0]:.1%})",
                yaxis_title=f"PC2 ({evr[1]:.1%})",
                zaxis_title=f"PC3 ({evr[2]:.1%})",
            ),
            height=600,
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("Not enough samples/dimensions for PCA 3D after filtering.")

    # --- Logreg ---
    acc, std_acc = logreg_cv(vectors_f, labels_f)
    st.subheader("Logistic regression CV accuracy (filtered)")
    st.caption(
        "Multinomial logistic regression on band-limited filtered radial profile vectors. "
        "5-fold stratified CV."
    )
    df_lr = pd.DataFrame(
        [
            {
                "Dataset": dataset_name,
                "Method": method,
                "Center %": center_percentage,
                "Freq band": f"{fmin}:{fmax}",
                "Drop %": drop_percentage,
                "Kept / Total": f"{kept_mask.sum()} / {len(kept_mask)}",
                "Mean accuracy": f"{acc:.4f}",
                "Std": f"{std_acc:.4f}",
            }
        ]
    )
    st.dataframe(df_lr, use_container_width=True)

    progress.progress(100, text="Done.")

    # --- Dropped thumbnails: side-by-side raw + processed ---
    fn = METHOD_FNS.get(method)
    n_dropped_total = int((~kept_mask).sum())
    if n_dropped_total > 0:
        st.subheader(f"Dropped images ({n_dropped_total} total)")
        for cls in class_names:
            dropped_indices = dropped_per_class.get(cls, [])
            if not dropped_indices:
                continue
            with st.expander(f"{cls} — {len(dropped_indices)} dropped"):
                show_indices = dropped_indices[:12]
                for idx in show_indices:
                    fname = filenames[idx]
                    img_path = image_dir / fname
                    dist_val = distances[idx]
                    col_raw, col_proc = st.columns(2)
                    with col_raw:
                        if img_path.exists():
                            st.image(
                                str(img_path),
                                caption=f"Raw  d={dist_val:.2f}",
                                width=150,
                            )
                        else:
                            st.text(f"Missing: {fname}")
                    with col_proc:
                        if img_path.exists():
                            raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                            if raw_img is not None and fn is not None:
                                proc_img = fn(raw_img)
                                st.image(
                                    proc_img,
                                    caption=f"Processed  d={dist_val:.2f}",
                                    width=150,
                                )
                            elif raw_img is not None:
                                st.image(
                                    raw_img,
                                    caption="(raw = processed for 'raw' method)",
                                    width=150,
                                )
                        else:
                            st.text("—")


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

    tab_cmp, tab_homo, tab_cls = st.tabs(
        ["Method comparison", "Homomorphic detail", "Class spectra"]
    )
    with tab_cmp:
        default_idx = files.index("_DSC1090.JPG") if "_DSC1090.JPG" in files else 0
        selected = st.selectbox("Select image", files, index=default_idx, key="img_cmp")
        image = cv2.imread(str(IMAGE_DIR / selected), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            render_comparison(image)
        else:
            st.error(f"Cannot read: {selected}")
    with tab_homo:
        default_idx = files.index("_DSC1090.JPG") if "_DSC1090.JPG" in files else 0
        selected = st.selectbox(
            "Select image", files, index=default_idx, key="img_homo"
        )
        image = cv2.imread(str(IMAGE_DIR / selected), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            render_homomorphic_detail(image)
        else:
            st.error(f"Cannot read: {selected}")
    with tab_cls:
        render_class_spectra_tab()


if __name__ == "__main__":
    main()
