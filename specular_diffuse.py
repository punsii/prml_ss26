"""
Specular/diffuse separation using chromaticity-based method.

Based on the dichromatic reflection model (Shafer 1985) and the iterative
separation approach of Tan & Ikeuchi (2005):
  "Separating Reflection Components of Textured Surfaces using a Single Image"

The dichromatic model states:
  I(x) = m_d(x) * c_d(x) + m_s(x) * c_s(x)

where:
  I    = observed image
  m_d  = diffuse magnitude (geometry-dependent)
  c_d  = diffuse chromaticity (material color)
  m_s  = specular magnitude
  c_s  = specular chromaticity (≈ illuminant color, often white)

Key insight: specular highlights shift pixel chromaticity toward the
illuminant color. By detecting and correcting this shift, we can separate
the specular component.

References:
  [1] Tan, R.T. & Ikeuchi, K. (2005). "Separating Reflection Components of
      Textured Surfaces using a Single Image." IEEE TPAMI 27(2), 178-193.
  [2] Yang, Q. et al. (2010). "Real-time Specular Highlight Removal Using
      Bilateral Filtering." ECCV 2010.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def estimate_illuminant(image: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Estimate illuminant color from brightest pixels (specular regions).

    Assumes specular highlights approximate the illuminant chromaticity.
    """
    gray = np.mean(image, axis=2)
    threshold = np.percentile(gray, percentile)
    bright_mask = gray >= threshold
    if bright_mask.sum() == 0:
        return np.array([1.0, 1.0, 1.0])
    illuminant = image[bright_mask].mean(axis=0)
    illuminant = illuminant / (np.linalg.norm(illuminant) + 1e-10)
    return illuminant


def compute_chromaticity(image: np.ndarray) -> np.ndarray:
    """Compute per-pixel chromaticity: c(x) = I(x) / sum(I(x))."""
    pixel_sum = image.sum(axis=2, keepdims=True)
    pixel_sum = np.maximum(pixel_sum, 1e-10)
    return image / pixel_sum


def separate_specular_diffuse(
    image: np.ndarray,
    illuminant: np.ndarray | None = None,
    iterations: int = 1,
    kernel_size: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate specular and diffuse components via iterative chromaticity.

    Uses the dichromatic reflection model: at each pixel the specular
    component shifts chromaticity toward the illuminant color.  We estimate
    the per-pixel specular magnitude from the minimum chromaticity ratio
    and subtract it iteratively.  A single bilateral filter pass at the end
    smooths the result.

    Args:
        image: Input BGR image (uint8 or float64 in [0,1]).
        illuminant: Illuminant color vector (3,). Estimated if None.
        iterations: Number of iterative refinement steps (default: 5).
        kernel_size: Kernel size for final bilateral filter (default: 15).

    Returns:
        diffuse: Diffuse component (same shape/dtype as input).
        specular: Specular component (same shape/dtype as input).
    """
    original_dtype = image.dtype
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    # Avoid zeros
    img = np.maximum(img, 1e-6)

    if illuminant is None:
        illuminant = estimate_illuminant(img)

    # Illuminant chromaticity
    illum_chroma = (illuminant / (illuminant.sum() + 1e-10)).reshape(1, 1, 3)

    # Pre-compute illuminant reciprocal for chromaticity ratio
    illum_chroma_inv = 1.0 / (illum_chroma + 1e-10)

    # Iterative specular subtraction — all ops stay in float32
    diffuse = img.copy()
    diff_intensity = np.empty((*img.shape[:2], 1), dtype=np.float32)

    for _ in range(iterations):
        np.sum(diffuse, axis=2, keepdims=True, out=diff_intensity)
        np.maximum(diff_intensity, 1e-10, out=diff_intensity)

        # Chromaticity ratio to illuminant; min across channels = specular fraction
        # Fused: min_c(diffuse_c / (intensity * illum_c)) * intensity * illum
        #      = min_c(diffuse_c / illum_c) * illum  (intensity cancels partially)
        ratio = diffuse * illum_chroma_inv  # (H,W,3)
        min_ratio = ratio.min(axis=2, keepdims=True)  # (H,W,1)
        specular_est = min_ratio * illum_chroma
        np.minimum(specular_est, diffuse, out=specular_est)
        np.subtract(diffuse, specular_est, out=diffuse)
        np.maximum(diffuse, 1e-6, out=diffuse)

    # Edge-preserving smooth via guided filter — O(n) per channel
    gray_guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    guided_radius = kernel_size // 2
    guided_eps = np.float32(0.01)
    for c in range(3):
        diffuse[:, :, c] = cv2.ximgproc.guidedFilter(
            guide=gray_guide,
            src=diffuse[:, :, c],
            radius=guided_radius,
            eps=guided_eps,
        )

    # Final specular = original - diffuse
    diffuse = np.clip(diffuse, 0, img.max())
    specular = np.clip(img - diffuse, 0, img.max())

    if original_dtype == np.uint8:
        diffuse = (diffuse * 255).clip(0, 255).astype(np.uint8)
        specular = (specular * 255).clip(0, 255).astype(np.uint8)

    return diffuse, specular


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate specular and diffuse reflection components"
    )
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Refinement iterations (default: 1)"
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=15,
        help="Bilateral filter kernel size (default: 15)",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    print(f"Processing {args.input} ({image.shape[1]}x{image.shape[0]})")
    diffuse, specular = separate_specular_diffuse(
        image,
        iterations=args.iterations,
        kernel_size=args.kernel_size,
    )

    diffuse_path = output_dir / f"{stem}_diffuse.png"
    specular_path = output_dir / f"{stem}_specular.png"
    cv2.imwrite(str(diffuse_path), diffuse)
    cv2.imwrite(str(specular_path), specular)
    print(f"Diffuse:  {diffuse_path}")
    print(f"Specular: {specular_path}")


if __name__ == "__main__":
    main()
