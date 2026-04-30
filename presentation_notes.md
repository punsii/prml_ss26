# Presentation Notes — Reflection Removal for Metal Surface Grading

## Format suggestions (LLM-friendly slide formats)

All of these take a single text file as input, so an LLM can author or revise the deck end-to-end.

- **Marp** — plain markdown with a YAML header (`marp: true`). Rendered to PDF / HTML / PPTX via `marp-cli`. One file, no JS, easiest to iterate. Recommended starting point.
- **Slidev** — Vue-based, polished transitions and syntax highlighting. Still markdown-driven, but heavier toolchain (`npx slidev`).
- **Quarto** (`.qmd`) — markdown plus runnable code blocks; renders via reveal.js. Useful if you want to embed live Python plots.
- **Pandoc + Beamer** — for LaTeX-style academic decks. Verbose, but classic.

If unsure: start with Marp, switch to Slidev only if the visuals matter.

## Slide outline

### 1. Problem

- Machine grinds/polishes metal across 4 stages: initial → grinding started → grinding decent → polished.
- A vision classifier grades the resulting surface finish.
- Strong specular reflections from workshop lighting obscure the surface texture and degrade classifier accuracy.
- Goal: reflection-removal as a preprocessing step → cleaner data → smaller / faster classifier (production motivation: total inference time = preprocess + classify).

### 2. Methods evaluated

| Method              | Domain               | Core idea                                                  |
| ------------------- | -------------------- | ---------------------------------------------------------- |
| CLAHE               | Spatial              | Tile-based histogram equalization with contrast clipping   |
| Multi-Scale Retinex | Log-spatial          | Subtract Gaussian-blurred illumination at multiple sigmas  |
| Homomorphic         | Log-frequency        | Gaussian high-pass on log(image) in the FFT domain         |
| Specular/Diffuse    | Color (chromaticity) | Dichromatic model — dropped (we go grayscale, see slide 4) |

CLAHE is contrast redistribution, not reflection removal — clipped specular pixels stay clipped. The other two physically model illumination × reflectance.

### 3. Patch-mode pipeline

- Input is 3648×3648. Features of interest sit at ~52 px (≈ 1/70 image size).
- Split into a 9×9 grid; process the 9 inner tiles at row/col indices (1, 4, 7).
- Outer tiles act as buffer so kernels never cross into other selected tiles → no seam artifacts.
- The same raster is intended for the classifier itself, so we never downsample the 4K image.

### 4. Optimization — grayscale

- Metal is monochromatic; the signal is texture, not color.
- 3× less data, 3× fewer FFTs in homomorphic, no per-channel loops in retinex/CLAHE.
- CLAHE no longer needs the LAB round-trip.
- Specular/diffuse separation depends on chromaticity → drops out (acceptable, it was already not the front-runner).

### 5. Optimization — patch mode

- Full-image 3648² FFT vs nine 405² FFTs: ~11× cheaper for homomorphic.
- The relative kernel size grows: same sigma is more aggressive against per-tile illumination.
- Patch-mode timing (per 9-tile batch) is what's reported in the streamlit app.

### 6. Frequency analysis: 2D → 1D

- Direction of grinding marks isn't needed for grading; only the magnitude per period matters.
- Squash the 2D FFT into a **radial profile**: mean magnitude per integer radius bin.
- Math: `profile[k] = mean( |F(u, v)| : k ≤ √(u² + v²) < k+1 )`
- Bin `k` ↔ spatial period `N / k` pixels for an `N × N` patch.
- Yields a 1D feature vector per tile. Peak position = mark spacing, peak height = mark prominence.

### 7. Two FFT artifacts, two independent fixes

- **Boundary cross artifact**: the 2D FFT treats the tile as periodic. Left edge ≠ right edge → implicit step in the wrap-around → broadband leakage as a bright cross through the spectrum.
  - Fix: **Hann window**. Multiply the tile by a 2D cosine bell that fades to zero at the borders before FFT. The wrap-around becomes 0 = 0; the cross disappears. Cost: O(N²) multiply, free vs the FFT.
- **Anisotropic radial sampling**: beyond `r = N/2`, only the corners of the FFT square contribute to the annulus → angular bias toward 45°.
  - Fix: **crop the radial profile at `r = N/2`**. The high frequencies we lose aren't relevant for our scale anyway.
- The two fixes solve different problems. Cropping does not substitute for windowing.

### 8. Visualization

- Linear y on the radial profile: DC and the next few bins dominate → trim first 3 bins.
- Log y (`log1p`) is cleaner; final app uses log scale.
- Streamlit app has two tabs: side-by-side method comparison, and a homomorphic-only detail tab with all 9 tiles + their spectra.

### 9. Open trade-offs (future work)

- **CLAHE**: cheapest but doesn't physically separate illumination — clipped specular regions stay clipped.
- **Retinex**: tends to wash out. Per-channel min-max normalization is the culprit; replace with percentile or z-score normalization.
- **Homomorphic**: tends to look dark. `gamma_low = 0.3` is too aggressive on the base brightness.
- **Distribution-shape normalization** (forcing each tile to a Gaussian histogram) collapses some discriminative information; z-score is gentler.

### 10. Numbers to gather before the talk

- Per-method patch-mode timing on the target hardware (currently shown in the streamlit app).
- End-to-end inference time: preprocess + classifier, for each candidate classifier.
- Classifier accuracy with vs without preprocessing — premise of the whole pipeline.
- Whether the radial-profile peak position alone correlates with grinding stage (potential cheap classifier baseline).

## Next steps to investigate

- Increase tile size — current 405 px (1/9 of image) may be smaller than needed; larger tiles support larger sigmas and richer radial profiles per tile.
- Run FFT + radial spectrum directly on the full image, with no preprocessing, as a baseline. If the radial peak is already discriminative, preprocessing may be optional.
- Average the radial spectrum across the 9 tiles for a single image to get one robust per-image profile.
- Build per-class statistics on the radial spectra: aggregate profiles across all images in each grinding-state bucket and compare bucket means / variances. Goal: see whether grinding stage is separable directly in the radial-spectrum feature space.

## Chronological notes — what changed and why

- Started with full-image color processing (CLAHE / Retinex / Homomorphic / Specular-Diffuse), Streamlit comparison app on the BMW test image.
- Switched to **patch mode** (9 tiles from a 9×9 grid). Output is per-tile, no full-image reconstruction needed; no seam artifacts.
- Switched to **grayscale** end-to-end. Methods became 3× cheaper; specular/diffuse retired.
- Streamlit UI: added a homomorphic-detail tab with red rectangles marking selected tiles, then per-tile original/processed/spectrum grids. Centered standalone images at ~50% width, 3×3 grids at ~70%.
- Replaced 2D FFT thumbnails with **1D radial profiles** (Hann + radial average + crop + log).
- Added `treefmt` as a flake check so formatting is enforced repo-wide.
