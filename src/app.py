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
import numpy as np
import streamlit as st

from clahe import apply_clahe
from homomorphic import homomorphic_filter
from retinex import multi_scale_retinex

IMAGE_DIR = Path("data/BMW_25/Rohdaten/Erste Bearbeitungsstufe 10-17ym")

GRID_SIZE = 9
TILE_INDICES = (1, 4, 7)
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


def extract_patches(image: np.ndarray) -> list[np.ndarray]:
    """Split image into a 9x9 grid; return selected tiles in row-major order."""
    h, w = image.shape[:2]
    th, tw = h // GRID_SIZE, w // GRID_SIZE
    return [
        image[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
        for i in TILE_INDICES
        for j in TILE_INDICES
    ]


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


RADIAL_DC_TRIM = 3  # drop DC + first 2 bins; they dwarf the rest on a linear plot


def radial_profile(image: np.ndarray) -> np.ndarray:
    """Mean FFT magnitude per integer radius.

    Hann-windowed (kills the boundary cross) and cropped to the inscribed
    circle (keeps the radial average isotropic).
    """
    h, w = image.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(image.astype(np.float32) * win)))
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - cy, xx - cx).astype(np.int32)
    summed = np.bincount(r.ravel(), weights=mag.ravel())
    counts = np.bincount(r.ravel())
    return (summed / np.maximum(counts, 1))[: min(cy, cx)]


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
            st.line_chart(np.log1p(radial_profile(results[CENTER_IDX])[RADIAL_DC_TRIM:]))
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

    tab_cmp, tab_homo = st.tabs(["Method comparison", "Homomorphic detail"])
    with tab_cmp:
        render_comparison(image)
    with tab_homo:
        render_homomorphic_detail(image)


if __name__ == "__main__":
    main()
