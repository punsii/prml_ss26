"""Shared radial-spectrum helpers used by app.py and class_spectra.py."""

import numpy as np

GRID_SIZE = 9
TILE_INDICES = (1, 4, 7)

# Fixed wavelength axis (in pixels) used for patch-size-invariant spectra.
# Linear spacing 2..200 px inclusive (199 wavelengths).
WAVELENGTHS = np.arange(2, 201, dtype=np.float64)

# Minimum image size that can be sampled at every wavelength on WAVELENGTHS:
# wavelength λ requires the FFT to have at least one full cycle, so
# min(h, w) >= max(WAVELENGTHS).
MIN_IMAGE_SIZE = int(WAVELENGTHS.max())


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


def radial_profile_at_wavelengths(
    image: np.ndarray, wavelengths: np.ndarray = WAVELENGTHS
) -> np.ndarray:
    """Mean FFT magnitude resampled onto a fixed wavelength axis (in pixels).

    For an image of size (h, w), with N = min(h, w), native radial bin r
    corresponds to wavelength N / r pixels. This function evaluates the
    native radial profile at the requested wavelengths via linear
    interpolation, making the result patch-size invariant: the same
    wavelength means the same physical frequency on patches of any size.

    Caller is responsible for ensuring min(h, w) >= max(wavelengths). Smaller
    images would silently edge-clamp at the Nyquist end, which numpy's interp
    does without warning.
    """
    profile = radial_profile(image)
    n = float(min(image.shape[:2]))
    target_bins = n / np.asarray(wavelengths, dtype=np.float64)
    bin_axis = np.arange(len(profile), dtype=np.float64)
    return np.interp(target_bins, bin_axis, profile)


