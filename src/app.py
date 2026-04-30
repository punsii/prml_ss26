"""Streamlit app: compare brightness normalization methods in patch mode.

Each image is split into a 9x9 grid; the 9 tiles at row/col indices (1, 4, 7)
are processed independently. Outer tiles are unused so kernels on the
selected tiles never need padding from outside the image.
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
    "CLAHE": apply_clahe,
    "Retinex": multi_scale_retinex,
    "Homomorphic": homomorphic_filter,
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


def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """Log-magnitude FFT spectrum of a BGR image (as grayscale)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    log_mag = np.log1p(magnitude)
    return (log_mag / log_mag.max() * 255).astype(np.uint8)


def render_comparison(image: np.ndarray) -> None:
    """Side-by-side comparison of all methods on the center tile."""
    original_center = extract_patches(image)[CENTER_IDX]
    original_rgb = cv2.cvtColor(original_center, cv2.COLOR_BGR2RGB)

    cols = st.columns(4)
    cols[0].subheader("Original (center)")
    cols[1].subheader("Result (center)")
    cols[2].subheader("FFT (result)")
    cols[3].subheader("Time (9 tiles)")

    for name, func in METHODS.items():
        results, elapsed = run_on_patches(func, image)
        center_rgb = cv2.cvtColor(results[CENTER_IDX], cv2.COLOR_BGR2RGB)
        fft = compute_fft_magnitude(results[CENTER_IDX])

        cols = st.columns(4)
        cols[0].image(original_rgb, caption="Original", use_container_width=True)
        cols[1].image(center_rgb, caption=name, use_container_width=True)
        cols[2].image(fft, caption=f"FFT ({name})", use_container_width=True)
        cols[3].metric(name, f"{elapsed:.3f}s")


def render_homomorphic_detail(image: np.ndarray) -> None:
    """Full image plus original/processed pair per tile, in 3x3 spatial layout."""
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
             caption="Full original", use_container_width=True)

    patches = extract_patches(image)
    results, elapsed = run_on_patches(homomorphic_filter, image)
    st.metric("Homomorphic — time (9 tiles)", f"{elapsed:.3f}s")

    n = len(TILE_INDICES)
    for row in range(n):
        cols = st.columns(n)
        for col in range(n):
            idx = row * n + col
            i, j = TILE_INDICES[row], TILE_INDICES[col]
            with cols[col]:
                st.caption(f"tile ({i}, {j})")
                st.image(cv2.cvtColor(patches[idx], cv2.COLOR_BGR2RGB),
                         caption="original", use_container_width=True)
                st.image(cv2.cvtColor(results[idx], cv2.COLOR_BGR2RGB),
                         caption="processed", use_container_width=True)


def main():
    st.set_page_config(layout="wide", page_title="Reflection Removal Comparison")
    st.title("Brightness Normalization Comparison (patch mode)")

    if not IMAGE_DIR.exists():
        st.error(f"Image directory not found: `{IMAGE_DIR}`")
        st.info("Run this app from the repo root directory.")
        return

    files = sorted(f.name for f in IMAGE_DIR.iterdir() if f.suffix.upper() in (".JPG", ".PNG"))
    if not files:
        st.error("No images found.")
        return

    default_idx = files.index("_DSC1090.JPG") if "_DSC1090.JPG" in files else 0
    selected = st.selectbox("Select image", files, index=default_idx)
    image = cv2.imread(str(IMAGE_DIR / selected))
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
