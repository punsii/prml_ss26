"""
Homomorphic filtering for local brightness normalization.

Models an image as illumination × reflectance.  In the log domain this
becomes a sum, so a high-pass filter in the frequency domain attenuates
the slowly-varying illumination component while preserving reflectance
(surface texture).

References:
  [1] Gonzalez, R.C. & Woods, R.E. (2008). "Digital Image Processing."
      Chapter 4.9 — Homomorphic Filtering.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def homomorphic_filter(
    image: np.ndarray,
    gamma_low: float = 0.3,
    gamma_high: float = 1.5,
    cutoff: float = 30.0,
) -> np.ndarray:
    """Apply homomorphic filtering to a BGR uint8 image.

    Args:
        image: Input BGR image (uint8).
        gamma_low: Gain for low frequencies (illumination). <1 suppresses.
        gamma_high: Gain for high frequencies (reflectance). >1 enhances.
        cutoff: Filter cutoff frequency in pixels.

    Returns:
        Filtered BGR image (uint8).
    """
    img = image.astype(np.float32) + 1.0
    log_img = np.log(img)

    # Build Gaussian high-pass filter in frequency domain
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    d_sq = u * u + v * v
    # H(u,v) = (gamma_high - gamma_low) * (1 - exp(-D²/2c²)) + gamma_low
    h = (gamma_high - gamma_low) * (1.0 - np.exp(-d_sq / (2.0 * cutoff * cutoff))) + gamma_low

    result = np.empty_like(img)
    for c in range(3):
        # DFT, shift, filter, inverse
        dft = np.fft.fft2(log_img[:, :, c])
        dft_shifted = np.fft.fftshift(dft)
        filtered = dft_shifted * h
        ch = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))
        # Back from log domain
        ch = np.exp(ch) - 1.0
        # Normalize to [0, 255]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-10) * 255.0
        result[:, :, c] = ch

    return result.clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply homomorphic filtering")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument(
        "--gamma-low", type=float, default=0.3, help="Low-freq gain (default: 0.3)"
    )
    parser.add_argument(
        "--gamma-high", type=float, default=1.5, help="High-freq gain (default: 1.5)"
    )
    parser.add_argument(
        "--cutoff", type=float, default=30.0, help="Filter cutoff freq (default: 30)"
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    print(f"Processing {args.input} ({image.shape[1]}x{image.shape[0]})")
    result = homomorphic_filter(
        image,
        gamma_low=args.gamma_low,
        gamma_high=args.gamma_high,
        cutoff=args.cutoff,
    )

    out_path = output_dir / f"{stem}_homomorphic.png"
    cv2.imwrite(str(out_path), result)
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
