"""
Multi-Scale Retinex (MSR) for local brightness normalization.

Retinex theory (Land 1977) models an image as illumination × reflectance.
By estimating illumination as a smoothed version of the log-image and
subtracting it, we recover the reflectance (surface detail) independent
of lighting.

Multi-Scale Retinex averages single-scale results at multiple Gaussian
sigma values for robustness across highlight sizes.

References:
  [1] Jobson, D.J. et al. (1997). "A Multiscale Retinex for Bridging the
      Gap Between Color Images and the Human Observation of Scenes."
      IEEE TIP 6(7), 965-976.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _fast_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with downsampling for large sigmas."""
    max_direct_sigma = 30.0
    if sigma <= max_direct_sigma:
        return cv2.GaussianBlur(img, (0, 0), sigma)

    # Downsample so effective sigma fits in max_direct_sigma
    scale = sigma / max_direct_sigma
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (0, 0), max_direct_sigma)
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)


def multi_scale_retinex(
    image: np.ndarray,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
) -> np.ndarray:
    """Apply Multi-Scale Retinex to a grayscale uint8 image.

    Args:
        image: Input grayscale image (uint8).
        sigmas: Gaussian blur sigmas for each scale.

    Returns:
        Normalized grayscale image (uint8).
    """
    img = image.astype(np.float32) + 1.0  # avoid log(0)
    log_img = np.log(img)

    retinex = np.zeros_like(log_img)
    for sigma in sigmas:
        blurred = _fast_gaussian(img, sigma)
        retinex += log_img - np.log(blurred + 1.0)
    retinex /= len(sigmas)

    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-10)
    return (retinex * 255).clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Multi-Scale Retinex")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[15.0, 80.0, 250.0],
        help="Gaussian sigmas (default: 15 80 250)",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    print(f"Processing {args.input} ({image.shape[1]}x{image.shape[0]})")
    result = multi_scale_retinex(image, sigmas=tuple(args.sigmas))

    out_path = output_dir / f"{stem}_retinex.png"
    cv2.imwrite(str(out_path), result)
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
