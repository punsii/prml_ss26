"""Streamlit app for comparing brightness normalization methods."""

import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import streamlit as st

from clahe import apply_clahe
from homomorphic import homomorphic_filter
from retinex import multi_scale_retinex

IMAGE_DIR = Path(
    "data/BMW_25/Rohdaten/Erste Bearbeitungsstufe 10-17ym"
)

import numpy as np

METHODS = {
    "CLAHE": lambda img: apply_clahe(img),
    "Retinex": lambda img: multi_scale_retinex(img),
    "Homomorphic": lambda img: homomorphic_filter(img),
}


def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute log-magnitude FFT spectrum of a BGR image (as grayscale)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dft = np.fft.fft2(gray)
    magnitude = np.abs(np.fft.fftshift(dft))
    log_mag = np.log1p(magnitude)
    log_mag = (log_mag / log_mag.max() * 255).astype(np.uint8)
    return log_mag


def run_method(name: str, func, image):
    t0 = time.perf_counter()
    result = func(image)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main():
    st.set_page_config(layout="wide", page_title="Reflection Removal Comparison")
    st.title("Brightness Normalization Comparison")

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
    image_path = IMAGE_DIR / selected
    image = cv2.imread(str(image_path))

    if image is None:
        st.error(f"Cannot read: {image_path}")
        return

    # Header row
    cols = st.columns(5)
    cols[0].subheader("Original")
    cols[1].subheader("FFT (original)")
    cols[2].subheader("Result")
    cols[3].subheader("FFT (result)")
    cols[4].subheader("Time")

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run all methods and compute FFTs once
    fft_original, fft_original_time = run_method("FFT (original)", compute_fft_magnitude, image)
    results = []
    for name, func in METHODS.items():
        result, elapsed = run_method(name, func, image)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        fft_result, fft_time = run_method(f"FFT ({name})", compute_fft_magnitude, result)
        results.append((name, result_rgb, fft_result, elapsed, fft_time))

    st.caption(f"FFT computation: {fft_original_time:.2f}s (original) · "
              + " · ".join(f"{fft_time:.2f}s ({name})" for name, _, _, _, fft_time in results))

    for name, result_rgb, fft_result, elapsed, _ in results:
        cols = st.columns(5)
        cols[0].image(original_rgb, caption="Original", use_container_width=True)
        cols[1].image(fft_original, caption="FFT (original)", use_container_width=True)
        cols[2].image(result_rgb, caption=name, use_container_width=True)
        cols[3].image(fft_result, caption=f"FFT ({name})", use_container_width=True)
        cols[4].metric(name, f"{elapsed:.2f}s")


if __name__ == "__main__":
    main()
