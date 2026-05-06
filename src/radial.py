"""Shared radial-spectrum helpers used by app.py and class_spectra.py."""

import numpy as np

GRID_SIZE = 9
TILE_INDICES = (1, 4, 7)
RADIAL_DC_TRIM = 3


def extract_patches(image: np.ndarray, scale: float = 1.0) -> list[np.ndarray]:
    """Extract 9 tiles centered at grid positions (1,4,7) of a GRID_SIZE x GRID_SIZE grid.

    tile_size = int(N // GRID_SIZE * scale), clamped so tiles don't exceed image bounds.
    Centers are at pixel (i + 0.5) * N / GRID_SIZE for i in TILE_INDICES.
    scale=1.0 reproduces original behaviour (tile_size = N // GRID_SIZE).
    """
    h, w = image.shape[:2]
    th_base = h // GRID_SIZE
    tw_base = w // GRID_SIZE
    th = int(th_base * scale)
    tw = int(tw_base * scale)

    patches = []
    for i in TILE_INDICES:
        cy = int((i + 0.5) * h / GRID_SIZE)
        for j in TILE_INDICES:
            cx = int((j + 0.5) * w / GRID_SIZE)
            r0 = max(0, cy - th // 2)
            r1 = min(h, r0 + th)
            r0 = max(0, r1 - th)
            c0 = max(0, cx - tw // 2)
            c1 = min(w, c0 + tw)
            c0 = max(0, c1 - tw)
            patches.append(image[r0:r1, c0:c1])
    return patches


def radial_profile(image: np.ndarray) -> np.ndarray:
    """Mean FFT magnitude per integer radius, Hann-windowed, cropped to inscribed circle."""
    h, w = image.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(image.astype(np.float32) * win)))
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - cy, xx - cx).astype(np.int32)
    summed = np.bincount(r.ravel(), weights=mag.ravel())
    counts = np.bincount(r.ravel())
    return (summed / np.maximum(counts, 1))[: min(cy, cx)]


def _radial_from_mag(mag: np.ndarray) -> np.ndarray:
    """Mean magnitude per integer radius from an already-fftshifted 2D magnitude array."""
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - cy, xx - cx).astype(np.int32)
    summed = np.bincount(r.ravel(), weights=mag.ravel())
    counts = np.bincount(r.ravel())
    return (summed / np.maximum(counts, 1))[: min(cy, cx)]


