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
from datetime import datetime
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
from percentile_model import build_model, load_model, mad_loo_cv, save_model
from homomorphic import homomorphic_filter
from radial import (GRID_SIZE, MIN_IMAGE_SIZE, TILE_INDICES, WAVELENGTHS,
                    _radial_from_mag, extract_patches,
                    radial_profile_at_wavelengths)
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
    # Thickness scales with image size so it reads consistently across datasets
    # (e.g. 3648 px BMW images vs ~1k Labor images).
    thickness = max(2, min(h, w) // 300)
    for i in TILE_INDICES:
        for j in TILE_INDICES:
            cv2.rectangle(
                out,
                (j * tw, i * th),
                ((j + 1) * tw, (i + 1) * th),
                color=(0, 0, 255),
                thickness=thickness,
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
    """3x3 grid of radial-spectrum line charts on the wavelength axis."""
    st.subheader(label)
    n = len(TILE_INDICES)
    wl_idx = pd.Index(WAVELENGTHS, name="Wavelength (px)")
    for row in range(n):
        _, mid, _ = st.columns([1, 5, 1])
        with mid:
            cols = st.columns(n)
            for col in range(n):
                df = pd.DataFrame(
                    {"log magnitude": np.log1p(profiles[row * n + col])},
                    index=wl_idx,
                )
                cols[col].line_chart(df)


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
            profile_wl = radial_profile_at_wavelengths(results[CENTER_IDX])
            st.line_chart(
                pd.DataFrame(
                    {"log magnitude": np.log1p(profile_wl)},
                    index=pd.Index(WAVELENGTHS, name="Wavelength (px)"),
                )
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
        [radial_profile_at_wavelengths(r) for r in results],
        "Radial spectrum (processed)",
    )


DATA_DIR = Path(os.environ.get("IMAGE_DIR", "data"))
LABOR_DIR = DATA_DIR / "Datensatz_Labor"
LABOR_CSV = DATA_DIR / "Labels_Datensatz_Labr.csv"
BMW_DIR = DATA_DIR / "BMW_25/Rohdaten"


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

MODELS_DIR = Path("models")


def _select_image(
    key_prefix: str, dataset_name: str | None = None
) -> tuple[str, str, Path] | None:
    """Three-tier image selector: dataset → class → image filename.

    If `dataset_name` is provided, the dataset selector is skipped (useful when
    a dataset is already chosen elsewhere in the same tab). Returns the
    selected (dataset_name, class_name, image_path) or None when no valid
    selection can be made.
    """
    if dataset_name is None:
        dataset_name = st.selectbox(
            "Dataset",
            list(DATASETS.keys()),
            key=f"{key_prefix}_dataset",
        )
    cfg = DATASETS[dataset_name]
    image_dir: Path = cfg["image_dir"]
    if not image_dir.exists():
        st.warning(f"Dataset directory not found: `{image_dir}`")
        return None
    if cfg["type"] == "csv":
        if not cfg["csv_path"].exists():
            st.warning(f"Labels CSV not found: `{cfg['csv_path']}`")
            return None
        rows = load_labels(cfg["csv_path"])
    else:
        rows = load_labels_from_dirs(image_dir)

    class_map: dict[str, list[str]] = {}
    for fname, label in rows:
        if (image_dir / fname).exists():
            class_map.setdefault(label, []).append(fname)
    if not class_map:
        st.warning("No labelled images found.")
        return None

    classes = sorted(class_map.keys())
    cls_name = st.selectbox("Class", classes, key=f"{key_prefix}_class")
    fnames = class_map[cls_name]
    fname = st.selectbox("Image", fnames, key=f"{key_prefix}_image")
    return dataset_name, cls_name, image_dir / fname


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
def _cached_vectors(dataset_name: str, method: str) -> dict | None:
    """Cache-keyed on (dataset_name, method). Returns raw vector data."""
    rows = _cached_rows(dataset_name)
    if rows is None:
        return None
    cfg = DATASETS[dataset_name]
    fn = METHOD_FNS.get(method)
    vectors, labels, filenames = compute_full_image_vectors(
        rows, cfg["image_dir"], fn
    )
    if len(vectors) == 0:
        return None
    return {"vectors": vectors, "labels": labels, "filenames": filenames}


def _load_vectors_with_progress(dataset_name: str, method: str) -> dict | None:
    """Load vectors with a real progress bar on first run; instant on cache hit."""
    cache_key = f"vectors_{dataset_name}_{method}"
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
        if min(img.shape[:2]) < MIN_IMAGE_SIZE:
            return None
        if fn is not None:
            img = fn(img)
        return (fname, label, radial_profile_at_wavelengths(img))

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


def _render_model_heatmap(
    model: dict[str, np.ndarray],
    wl_axis: list[float],
    title: str,
) -> None:
    """Per-class percentile-model heatmap with equipotential contour lines.

    Uses go.Contour with `coloring='heatmap'` so the gradient fill matches the
    old heatmap look while contour lines make small inter-class differences
    visible at a glance.
    """
    if not model:
        st.info("Empty model — nothing to render.")
        return
    percentiles_axis = list(range(1, 101))
    log_arrs = {cls: np.log10(np.maximum(arr, 1e-6)) for cls, arr in model.items()}
    cls_list = sorted(log_arrs)
    zmin = float(min(log_arrs[c].min() for c in cls_list))
    zmax = float(max(log_arrs[c].max() for c in cls_list))
    # Even contour spacing across classes so lines are directly comparable.
    n_contours = 18
    contour_step = (zmax - zmin) / n_contours if zmax > zmin else 1.0

    fig = make_subplots(
        rows=1,
        cols=len(cls_list),
        subplot_titles=cls_list,
        shared_yaxes=True,
    )
    for col_i, cls in enumerate(cls_list, start=1):
        fig.add_trace(
            go.Contour(
                z=log_arrs[cls].tolist(),
                x=percentiles_axis,
                y=wl_axis,
                colorscale="Viridis",
                zmin=zmin,
                zmax=zmax,
                contours=dict(
                    coloring="heatmap",
                    showlines=True,
                    start=zmin,
                    end=zmax,
                    size=contour_step,
                ),
                line=dict(color="rgba(0,0,0,0.45)", width=0.6),
                showscale=(col_i == len(cls_list)),
                colorbar=dict(title="log₁₀ magnitude"),
                hovertemplate="Percentile: %{x}<br>λ: %{y} px<br>log₁₀ mag: %{z:.2f}<extra></extra>",
            ),
            row=1,
            col=col_i,
        )
        fig.update_xaxes(title_text="Percentile", row=1, col=col_i)
    fig.update_yaxes(title_text="Wavelength (px)", row=1, col=1)
    fig.update_layout(title=title, height=500)
    st.plotly_chart(fig, width="stretch")


def _render_fft_explorer(dataset_name: str, method: str) -> None:
    """Show 2D FFT magnitude + radial profile for a single selected image."""
    selection = _select_image("fft_explorer", dataset_name=dataset_name)
    if selection is None:
        return
    _, _, selected_path = selection

    raw_img = cv2.imread(str(selected_path), cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        st.error(f"Cannot read: {selected_path}")
        return

    fn = METHOD_FNS.get(method)
    img = fn(raw_img) if fn is not None else raw_img

    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float32) * win)))

    col_orig, col_a, col_b = st.columns(3)
    with col_orig:
        st.image(raw_img, caption="Original", width="stretch")
    with col_a:
        log_mag = np.log1p(mag)
        fft_gray = (log_mag / log_mag.max() * 255).astype(np.uint8)
        st.image(fft_gray, caption="2D FFT magnitude (log)", width="stretch")
    with col_b:
        profile_wl = radial_profile_at_wavelengths(img)
        fig_rad = go.Figure(
            go.Scatter(
                x=WAVELENGTHS.tolist(),
                y=profile_wl.tolist(),
                mode="lines",
                name="magnitude",
            )
        )
        fig_rad.update_yaxes(type="log")
        fig_rad.update_layout(
            title="Radial profile (log y)",
            xaxis_title="Wavelength (px)",
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

    # --- 2D FFT explorer ---
    with st.expander("2D power spectrum explorer", expanded=False):
        _render_fft_explorer(dataset_name, method)

    drop_percentage = st.slider(
        "Drop percentage",
        0.0,
        1.0,
        0.1,
        0.01,
        help="Fraction of each class to drop as Mahalanobis outliers. Applied after the cached vector computation.",
    )

    # --- Saved models (load path) ---
    MODELS_DIR.mkdir(exist_ok=True)
    saved_files = sorted(
        MODELS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    with st.expander(f"Saved models ({len(saved_files)} in ./models/)", expanded=False):
        if not saved_files:
            st.caption(
                "No saved models. Run Compute and use 'Save model' to write one."
            )
        else:
            chosen = st.selectbox(
                "Saved model",
                saved_files,
                format_func=lambda p: p.name,
                key="saved_model_select",
            )
            if st.button("Load and show heatmap", key="load_model_btn"):
                try:
                    loaded_model, loaded_wl = load_model(chosen)
                except Exception as exc:
                    st.error(f"Failed to load: {exc}")
                else:
                    st.success(f"Loaded {chosen.name}")
                    _render_model_heatmap(
                        loaded_model,
                        loaded_wl.tolist(),
                        f"Percentile model — {chosen.stem}",
                    )

    # st.button returns True only on the rerun triggered by the click itself,
    # which would unmount the Compute output the moment the user clicked any
    # downstream widget (e.g. Save). Latch a flag in session_state instead so
    # the output sticks and downstream widgets re-render on subsequent reruns.
    if st.button("Compute"):
        st.session_state["spectra_computed"] = True

    if not st.session_state.get("spectra_computed"):
        st.info("Press 'Compute' to run the analysis.")
        return

    cached = _load_vectors_with_progress(dataset_name, method)

    if cached is None:
        st.error("No images found or data unavailable.")
        return

    vectors = cached["vectors"]
    labels = cached["labels"]
    filenames = cached["filenames"]

    wl_axis = WAVELENGTHS.tolist()

    with st.spinner("Filtering outliers…"):
        kept_mask, distances, dropped_per_class = mahalanobis_filter(
            vectors, labels, drop_percentage
        )

    vectors_f = vectors[kept_mask]
    labels_f = labels[kept_mask]

    class_names = sorted(np.unique(labels).tolist())
    class_stats_data = compute_class_stats(vectors_f, labels_f)

    # --- Build the model up-front so the Save button at the top has it ready ---
    model = build_model(vectors_f, labels_f)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_name = (
        f"{dataset_name.replace(' ', '_')}__{method}__"
        f"drop{drop_percentage:.2f}__{ts}.csv"
    )
    if st.button(f"Save model to ./models/{default_name}", key="save_model_btn"):
        out_path = MODELS_DIR / default_name
        save_model(model, WAVELENGTHS, out_path)
        st.success(f"Saved to {out_path}")

    # --- Frequency spectra (left) + per-class probability point cloud (right) ---
    fig_freq = go.Figure()
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
            x_all.extend(wl_axis)
            y_all.extend([y_norm] * len(wl_axis))
            z_all.extend(z_row.tolist())
            x_all.append(None)
            y_all.append(None)
            z_all.append(None)
        fig_freq.add_trace(
            go.Scatter3d(
                x=x_all,
                y=y_all,
                z=z_all,
                mode="lines",
                line=dict(color=color, width=2),
                name=cls,
            )
        )
    fig_freq.update_layout(
        title=f"Frequency spectra (per training image) — {method}",
        scene=dict(
            xaxis_title="Wavelength (px)",
            yaxis_title="Image index (normalised)",
            zaxis_title="log₁₀ FFT magnitude",
        ),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Histogram (freq_bin, log10 magnitude) per class; one Scatter3d marker per
    # non-empty cell at z = log(1 + count). Empty cells produce nothing, so
    # each cloud is naturally clipped to its own support.
    n_mag_bins = 60
    log_vecs = np.log10(np.maximum(vectors_f, 1e-6))
    log_min = float(log_vecs.min())
    log_max = float(log_vecs.max())
    mag_edges = np.linspace(log_min, log_max, n_mag_bins + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])

    fig_cloud = go.Figure()
    for cls in class_names:
        mask = labels_f == cls
        if not mask.any():
            continue
        color = colors.get(cls, "gray")
        cls_log = log_vecs[mask]
        n_wls = cls_log.shape[1]
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for w in range(n_wls):
            counts, _ = np.histogram(cls_log[:, w], bins=mag_edges)
            nonzero = np.nonzero(counts)[0]
            if nonzero.size == 0:
                continue
            xs.extend([wl_axis[w]] * nonzero.size)
            ys.extend(mag_centers[nonzero].tolist())
            zs.extend(np.log1p(counts[nonzero]).tolist())
        fig_cloud.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.6),
                name=cls,
            )
        )
    fig_cloud.update_layout(
        title=f"Per-class probability density — {method}",
        scene=dict(
            xaxis_title="Wavelength (px)",
            yaxis_title="log₁₀ magnitude",
            zaxis_title="log(1 + count)",
        ),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    col_freq, col_cloud = st.columns(2)
    with col_freq:
        st.plotly_chart(fig_freq, width="stretch")
    with col_cloud:
        st.plotly_chart(fig_cloud, width="stretch")

    # --- Percentile model heatmap with contour lines ---
    _render_model_heatmap(
        model, wl_axis, f"Percentile model — {dataset_name} — {method}"
    )

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
        "MAD: percentile model, leave-one-out CV, MAD-from-50 scoring."
    )
    shared = {
        "Dataset": dataset_name,
        "Method": method,
        "Drop %": drop_percentage,
        "Kept / Total": f"{kept_mask.sum()} / {len(kept_mask)}",
    }
    st.dataframe(
        pd.DataFrame([
            {**shared, "Classifier": "Logistic regression (5-fold)", "Accuracy": f"{acc_lr:.4f}", "Std": f"{std_lr:.4f}"},
            {**shared, "Classifier": "MAD percentile model (LOO)",   "Accuracy": f"{acc_mad:.4f}", "Std": "—"},
        ]),
        width="stretch",
    )

    # --- Dropped thumbnails: raw (+ processed if not raw method) + FFT ---
    fn = METHOD_FNS.get(method)
    is_raw_method = fn is None
    n_dropped_total = int((~kept_mask).sum())
    if n_dropped_total > 0:
        st.subheader(f"Dropped images ({n_dropped_total} total)")
        thumb_w = 420
        for cls in class_names:
            dropped_indices = dropped_per_class.get(cls, [])
            if not dropped_indices:
                continue
            st.markdown(f"**{cls}** — {len(dropped_indices)} dropped")
            for idx in dropped_indices[:12]:
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
                    st.image(raw_img, caption=f"Raw  d={dist_val:.2f}", width=thumb_w)
                if not is_raw_method:
                    with cols[1]:
                        st.image(
                            fn(raw_img),
                            caption=f"Processed  d={dist_val:.2f}",
                            width=thumb_w,
                        )
                with cols[1 if is_raw_method else 2]:
                    h, w = raw_img.shape
                    win = np.outer(np.hanning(h), np.hanning(w))
                    mag = np.abs(np.fft.fftshift(
                        np.fft.fft2(raw_img.astype(np.float32) * win)
                    ))
                    log_mag = np.log1p(mag)
                    fft_img = (log_mag / log_mag.max() * 255).astype(np.uint8)
                    st.image(fft_img, caption="FFT magnitude (log)", width=thumb_w)


def main():
    st.set_page_config(layout="wide", page_title="Reflection Removal Comparison")
    st.title("Brightness Normalization Comparison (patch mode, grayscale)")

    tab_cmp, tab_homo, tab_cls = st.tabs(
        ["Method comparison", "Homomorphic detail", "Class spectra"]
    )
    with tab_cmp:
        selection = _select_image("img_cmp")
        if selection is not None:
            _, _, path = selection
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                render_comparison(image)
            else:
                st.error(f"Cannot read: {path}")
    with tab_homo:
        selection = _select_image("img_homo")
        if selection is not None:
            _, _, path = selection
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                render_homomorphic_detail(image)
            else:
                st.error(f"Cannot read: {path}")
    with tab_cls:
        render_class_spectra_tab()


if __name__ == "__main__":
    main()
