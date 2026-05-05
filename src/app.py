"""Streamlit app: compare brightness normalization methods in patch mode (grayscale).

Each image is loaded as grayscale and split into a 9x9 grid; the 9 tiles at
row/col indices (1, 4, 7) are processed independently. Outer tiles are
unused so kernels on the selected tiles never need padding from outside
the image.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure src/ is on sys.path when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clahe import apply_clahe
from class_stats import (CLASS_COLORS, compute_class_stats,
                         compute_full_image_vectors, load_labels,
                         load_labels_from_dirs, logreg_cv, mahalanobis_filter)
from percentile_atlas import build_atlas, mad_loo_cv, rank_profiles, score_spectrum
from homomorphic import homomorphic_filter
from radial import (GRID_SIZE, RADIAL_DC_TRIM, TILE_INDICES, _radial_from_mag,
                    center_crop, extract_patches, radial_profile,
                    radial_profile_center_fraction)
from retinex import multi_scale_retinex

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
                cols[col].image(images[row * n + col], width="stretch")


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
            width="stretch",
        )

        cols = st.columns(4)
        cols[0].image(original_center, caption="Original (center)", width="stretch")
        cols[1].image(results[CENTER_IDX], caption=f"{name} (center)", width="stretch")
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
        width="stretch",
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
    "Step_0_17-25ym": "red",
    "Step_1_10-17ym": "orange",
    "Step_2_3-10ym": "green",
    "Step_3_final_3ym": "blue",
}

METHOD_FNS = {
    "raw": None,
    "clahe": apply_clahe,
    "retinex": multi_scale_retinex,
    "homomorphic": homomorphic_filter,
}


@st.cache_data(show_spinner=False)
def _cached_rows(dataset_name: str) -> list[tuple[str, str]] | None:
    """Cache label rows for a dataset (cheap — just reads CSV or directory listing)."""
    cfg = DATASETS[dataset_name]
    image_dir = cfg["image_dir"]
    if not image_dir.exists():
        return None
    if cfg["type"] == "csv":
        csv_path = cfg["csv_path"]
        if not csv_path.exists():
            return None
        return load_labels(csv_path)
    return load_labels_from_dirs(image_dir)


@st.cache_data(show_spinner=False)
def _cached_vectors(
    dataset_name: str, method: str, center_percentage: float
) -> dict | None:
    """Cache-keyed on (dataset_name, method, center_percentage). Returns raw vector data."""
    rows = _cached_rows(dataset_name)
    if rows is None:
        return None
    cfg = DATASETS[dataset_name]
    fn = METHOD_FNS.get(method)
    vectors, labels, filenames = compute_full_image_vectors(
        rows, cfg["image_dir"], fn, center_percentage
    )
    if len(vectors) == 0:
        return None
    return {"vectors": vectors, "labels": labels, "filenames": filenames}


def _load_vectors_with_progress(
    dataset_name: str, method: str, center_percentage: float
) -> dict | None:
    """Load vectors with a real progress bar on first run; instant on cache hit."""
    cache_key = f"vectors_{dataset_name}_{method}_{center_percentage}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    rows = _cached_rows(dataset_name)
    if rows is None:
        return None

    cfg = DATASETS[dataset_name]
    image_dir = cfg["image_dir"]
    fn = METHOD_FNS.get(method)

    n_workers = min(8, (os.cpu_count() or 1))
    n = len(rows)
    progress = st.progress(0, text=f"Loading images… 0/{n}")

    def _process_one(args: tuple) -> tuple | None:
        fname, label = args
        path = image_dir / fname
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if fn is not None:
            img = fn(img)
        return (fname, label, radial_profile_center_fraction(img, center_percentage))

    # Submit all tasks; collect results in completion order for progress, then
    # re-sort by original index to preserve row order for reproducibility.
    ordered: dict[int, tuple] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_to_idx = {
            pool.submit(_process_one, row): i for i, row in enumerate(rows)
        }
        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result is not None:
                ordered[idx] = result
            done_count += 1
            pct = int(done_count / n * 90)
            progress.progress(pct, text=f"Loading images… {done_count}/{n}")

    profiles: list[np.ndarray] = []
    labels_list: list[str] = []
    filenames_list: list[str] = []
    for idx in sorted(ordered):
        fname, label, profile = ordered[idx]
        filenames_list.append(fname)
        labels_list.append(label)
        profiles.append(profile)

    progress.progress(95, text="Assembling vectors…")

    if not profiles:
        progress.empty()
        return None

    min_len = min(len(p) for p in profiles)
    vectors = np.array([p[:min_len] for p in profiles])
    result = {
        "vectors": vectors,
        "labels": np.array(labels_list),
        "filenames": filenames_list,
    }

    progress.progress(100, text="Done.")
    progress.empty()

    # Also populate st.cache_data so future calls via _cached_vectors are instant
    st.session_state[cache_key] = result
    return result


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

    if cfg["type"] == "dirs":
        labels_display = [f"[{p.parent.name}] {p.name}" for p in img_paths]
    else:
        labels_display = [str(p.relative_to(image_dir)) for p in img_paths]
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
        st.image(raw_img, caption="Original", width="stretch")
    with col_a:
        log_mag = np.log1p(mag)
        fft_gray = (log_mag / log_mag.max() * 255).astype(np.uint8)
        st.image(fft_gray, caption="2D FFT magnitude (log)", width="stretch")
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
        st.plotly_chart(fig_rad, width="stretch")


def render_class_spectra_tab() -> None:
    st.header("Class spectra")

    # --- Dataset selector ---
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), index=1)
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

    cached = _load_vectors_with_progress(dataset_name, method, center_percentage)

    if cached is None:
        st.error("No images found or data unavailable.")
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
        return

    vectors_b = vectors[:, fmin:fmax]
    bin_axis = list(range(fmin, fmax))

    with st.spinner("Filtering outliers…"):
        kept_mask, distances, dropped_per_class = mahalanobis_filter(
            vectors_b, labels, drop_percentage
        )

    vectors_f = vectors_b[kept_mask]
    labels_f = labels[kept_mask]

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
    st.plotly_chart(fig_spec, width="stretch")

    # --- Plot 1b: Waterfall spectra ---
    with st.expander("Waterfall spectra (per-image 3D)", expanded=False):
        fig_wf = go.Figure()
        for cls in class_names:
            mask = labels_f == cls
            if not mask.any():
                continue
            color = colors.get(cls, "gray")
            cls_vecs = vectors_f[mask]
            n_imgs = len(cls_vecs)
            x_all: list = []
            y_all: list = []
            z_all: list = []
            for i, row in enumerate(cls_vecs):
                y_norm = i / max(n_imgs - 1, 1)
                z_row = np.log10(np.maximum(row, 1e-6))
                x_all.extend(bin_axis)
                y_all.extend([y_norm] * len(bin_axis))
                z_all.extend(z_row.tolist())
                x_all.append(None)
                y_all.append(None)
                z_all.append(None)
            fig_wf.add_trace(
                go.Scatter3d(
                    x=x_all,
                    y=y_all,
                    z=z_all,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=cls,
                )
            )
        fig_wf.update_layout(
            title=f"Waterfall spectra — {dataset_name} — {method}  [bins {fmin}:{fmax}]",
            scene=dict(
                xaxis_title="Radial bin",
                yaxis_title="Image index (normalised)",
                zaxis_title="log₁₀ FFT magnitude",
            ),
            height=700,
        )
        st.plotly_chart(fig_wf, width="stretch")

    # --- Plot 1c: Percentile atlas ---
    with st.expander("Percentile atlas (per-class)", expanded=False):
        percentiles = np.arange(1, 101)
        atlas: dict[str, np.ndarray] = {}
        for cls in class_names:
            mask = labels_f == cls
            if not mask.any():
                continue
            cls_vecs = vectors_f[mask]
            # percentile() returns (100, n_bins); transpose to (n_bins, 100)
            atlas[cls] = np.log10(np.maximum(
                np.percentile(cls_vecs, percentiles, axis=0), 1e-6
            )).T

        if atlas:
            cls_list = [c for c in class_names if c in atlas]
            zmin = float(min(v.min() for v in atlas.values()))
            zmax = float(max(v.max() for v in atlas.values()))

            fig_atl = make_subplots(
                rows=1,
                cols=len(cls_list),
                subplot_titles=cls_list,
                shared_yaxes=True,
            )
            for col_i, cls in enumerate(cls_list, start=1):
                fig_atl.add_trace(
                    go.Heatmap(
                        z=atlas[cls].tolist(),
                        x=percentiles.tolist(),
                        y=bin_axis,
                        colorscale="Viridis",
                        zmin=zmin,
                        zmax=zmax,
                        showscale=(col_i == len(cls_list)),
                        colorbar=dict(title="log₁₀ magnitude"),
                        hovertemplate="Percentile: %{x}<br>Bin: %{y}<br>log₁₀ mag: %{z:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=col_i,
                )
                fig_atl.update_xaxes(title_text="Percentile", row=1, col=col_i)
            fig_atl.update_yaxes(title_text="Radial bin", row=1, col=1)
            fig_atl.update_layout(
                title=f"Percentile atlas — {dataset_name} — {method}  [bins {fmin}:{fmax}]",
                height=500,
            )
            st.plotly_chart(fig_atl, width="stretch")

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
        st.plotly_chart(fig_pca, width="stretch")
    else:
        st.info("Not enough samples/dimensions for PCA 3D after filtering.")

    # --- CV accuracy comparison ---
    acc_lr, std_lr = logreg_cv(vectors_f, labels_f)
    with st.spinner("MAD LOO CV…"):
        acc_mad = mad_loo_cv(vectors_f, labels_f)
    st.subheader("Classification accuracy (filtered)")
    st.caption(
        "LR: multinomial logistic regression, 5-fold stratified CV.  "
        "MAD: percentile atlas, leave-one-out CV, MAD-from-50 scoring."
    )
    shared = {
        "Dataset": dataset_name,
        "Method": method,
        "Center %": center_percentage,
        "Freq band": f"{fmin}:{fmax}",
        "Drop %": drop_percentage,
        "Kept / Total": f"{kept_mask.sum()} / {len(kept_mask)}",
    }
    st.dataframe(
        pd.DataFrame([
            {**shared, "Classifier": "Logistic regression (5-fold)", "Accuracy": f"{acc_lr:.4f}", "Std": f"{std_lr:.4f}"},
            {**shared, "Classifier": "MAD percentile atlas (LOO)",   "Accuracy": f"{acc_mad:.4f}", "Std": "—"},
        ]),
        width="stretch",
    )

    # --- Classify new image ---
    atlas = build_atlas(vectors_f, labels_f)
    with st.expander("Classify new image", expanded=False):
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded is not None:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            query_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if query_raw is None:
                st.error("Could not decode image.")
            else:
                fn_q = METHOD_FNS.get(method)
                query_img = fn_q(query_raw) if fn_q is not None else query_raw
                q = radial_profile_center_fraction(query_img, center_percentage)[fmin:fmax]

                scores = score_spectrum(q, atlas)
                ranks = rank_profiles(q, atlas)
                pred_cls = max(scores, key=scores.__getitem__)

                st.success(f"Predicted class: **{pred_cls}**")

                fig_scores = go.Figure(go.Bar(
                    x=list(scores.keys()),
                    y=list(scores.values()),
                    marker_color=[colors.get(c, "gray") for c in scores],
                ))
                fig_scores.update_layout(
                    title="MAD-from-50 score per class (higher = better fit)",
                    yaxis_title="Score",
                    height=300,
                )
                st.plotly_chart(fig_scores, width="stretch")

                fig_ranks = go.Figure()
                fig_ranks.add_hline(y=50, line_dash="dash", line_color="gray",
                                    annotation_text="median")
                for cls in class_names:
                    if cls not in ranks:
                        continue
                    fig_ranks.add_trace(go.Scatter(
                        x=bin_axis,
                        y=ranks[cls].tolist(),
                        mode="lines",
                        name=cls,
                        line=dict(color=colors.get(cls, "gray")),
                    ))
                fig_ranks.update_layout(
                    title="Per-bin percentile rank within each class (50 = perfect median fit)",
                    xaxis_title="Radial bin",
                    yaxis_title="Percentile rank",
                    height=400,
                )
                st.plotly_chart(fig_ranks, width="stretch")

    # --- Dropped thumbnails: raw (+ processed if not raw method) + FFT ---
    fn = METHOD_FNS.get(method)
    is_raw_method = fn is None
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
                    cols = st.columns(2 if is_raw_method else 3)
                    raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if raw_img is None:
                        with cols[0]:
                            st.text(f"Missing: {fname}")
                        continue
                    with cols[0]:
                        st.image(raw_img, caption=f"Raw  d={dist_val:.2f}", width=280)
                    if not is_raw_method:
                        with cols[1]:
                            st.image(fn(raw_img), caption=f"Processed  d={dist_val:.2f}", width=280)
                    with cols[1 if is_raw_method else 2]:
                        cropped = center_crop(raw_img, center_percentage)
                        h, w = cropped.shape
                        win = np.outer(np.hanning(h), np.hanning(w))
                        mag = np.abs(np.fft.fftshift(
                            np.fft.fft2(cropped.astype(np.float32) * win)
                        ))
                        log_mag = np.log1p(mag)
                        fft_img = (log_mag / log_mag.max() * 255).astype(np.uint8)
                        st.image(fft_img, caption="FFT magnitude (log)", width=280)


def main():
    st.set_page_config(layout="wide", page_title="Reflection Removal Comparison")
    st.title("Brightness Normalization Comparison (patch mode, grayscale)")

    if not BMW_DIR.exists():
        st.error(f"Image directory not found: `{BMW_DIR}`")
        st.info("Run this app from the repo root directory.")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_entries = [
        (f"[{subdir.name}] {p.name}", p)
        for subdir in sorted(BMW_DIR.iterdir())
        if subdir.is_dir()
        for p in sorted(subdir.iterdir())
        if p.suffix.lower() in exts
    ]
    if not img_entries:
        st.error("No images found.")
        return

    display_names = [label for label, _ in img_entries]
    img_paths_map = {label: path for label, path in img_entries}

    tab_cmp, tab_homo, tab_cls = st.tabs(
        ["Method comparison", "Homomorphic detail", "Class spectra"]
    )
    with tab_cmp:
        selected = st.selectbox("Select image", display_names, index=0, key="img_cmp")
        image = cv2.imread(str(img_paths_map[selected]), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            render_comparison(image)
        else:
            st.error(f"Cannot read: {selected}")
    with tab_homo:
        selected = st.selectbox("Select image", display_names, index=0, key="img_homo")
        image = cv2.imread(str(img_paths_map[selected]), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            render_homomorphic_detail(image)
        else:
            st.error(f"Cannot read: {selected}")
    with tab_cls:
        render_class_spectra_tab()


if __name__ == "__main__":
    main()
