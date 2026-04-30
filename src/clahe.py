"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) for local
brightness normalization of metal surface images.

Operates on grayscale: divides image into tiles and equalizes histogram
per tile with contrast limiting.
"""

import argparse
from pathlib import Path

import cv2


def apply_clahe(
    image: cv2.typing.MatLike,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> cv2.typing.MatLike:
    """Apply CLAHE to a single-channel uint8 image."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply CLAHE brightness normalization")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument(
        "--clip-limit", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)"
    )
    parser.add_argument(
        "--tile-size", type=int, default=8, help="CLAHE tile grid size (default: 8)"
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    print(f"Processing {args.input} ({image.shape[1]}x{image.shape[0]})")
    result = apply_clahe(image, clip_limit=args.clip_limit, tile_size=args.tile_size)

    out_path = output_dir / f"{stem}_clahe.png"
    cv2.imwrite(str(out_path), result)
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
