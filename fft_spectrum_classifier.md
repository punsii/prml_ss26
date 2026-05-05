# FFT Spectrum Percentile Classifier

## Concept

For each class and each radial frequency bin, compute the full empirical distribution of FFT magnitudes across all training images. This gives a `(n_classes, n_bins, 100)` tensor of percentile values. When a new image arrives, its spectrum is compared against each class's percentile profile to produce a class score.

## Building the Percentile Atlas

1. For each class `c`, collect the `(N_c, n_bins)` matrix of training spectra (after Mahalanobis outlier removal, same as current pipeline).
2. For each bin `b`, compute `np.percentile(spectra[:, b], np.arange(1, 101))` → 100 values.
3. Result: `atlas[c][b][p]` = the magnitude that `p%` of class `c` images fall below at bin `b`.

## Scoring a New Spectrum

Given a query spectrum `q` (shape `(n_bins,)`), for each class `c`:

1. For each bin `b`, find the percentile rank `r_b` of `q[b]` within `atlas[c][b]` — i.e., how many of the 100 percentile thresholds `q[b]` exceeds. `r_b ∈ [0, 100]`.
2. This produces a rank vector `R_c` (shape `(n_bins,)`) of percentile positions within class `c`.
3. A spectrum that "fits" class `c` well will have ranks clustered near 50 (median); outliers will push ranks toward 0 or 100.

## Score Function (open problem)

Naively multiplying all per-bin probabilities together collapses to near-zero (curse of dimensionality). Options to explore:

- **Median absolute deviation from 50**: `score(c) = -mean(|R_c - 50|)`. Higher is better; a perfect median match scores 0.
- **Log-likelihood**: model each bin as independent Gaussian, compute log-likelihood of `q[b]` under `N(mean_c_b, std_c_b)`. Sum over bins. Already well-defined and avoids the product collapse.
- **Rank histogram entropy**: measure how uniformly `R_c` is distributed; a flat distribution (all ranks equally likely) indicates the query is indistinguishable from the training spread.
- **Mahalanobis distance**: the existing outlier filter already does this. Extend it to scoring: `score(c) = -mahalanobis(q, mean_c, cov_c)`.

The percentile approach and the Mahalanobis approach converge if the underlying distribution is Gaussian; percentiles are more robust to heavy-tailed or multimodal bins.

## Scale Matching (required before any of the above)

Camera resolution (pixel pitch, lens magnification) directly affects which physical spatial frequency maps to which radial bin. Two images of the same surface at different resolutions will produce non-overlapping spectra unless corrected.

Options:
- **Normalise by Nyquist**: express bins as fraction of max bin (`bin / n_bins`). This maps all spectra to [0, 1] independent of image size.
- **Physical frequency**: if pixel size (µm/px) is known per dataset, convert bins to cycles/mm. The BMW dataset metadata may contain this.
- **Reference peak alignment**: find a dominant spectral peak common to all classes (e.g., grinding line fundamental), align spectra by shifting to that peak. Brittle but works when the peak is consistent.

The simplest first step is Nyquist normalisation; it requires no metadata and is already implicit if all images in a dataset have the same resolution.

## Implementation Plan

1. **`src/percentile_atlas.py`** — new module
   - `build_atlas(vectors, labels) -> dict[str, np.ndarray]`  
     Returns `{class: (n_bins, 100)}` percentile arrays.
   - `score_spectrum(q, atlas) -> dict[str, float]`  
     Returns per-class score using MAD-from-50 (start simple, swap in later).
   - `predict(q, atlas) -> str`  
     Returns argmax class.

2. **`src/app.py`** — add to class-spectra tab
   - New expander: "Percentile atlas" — heatmap of `atlas[c]` (bins × percentiles) for each class, using Plotly `go.Heatmap`. This directly visualises the distribution shape at every frequency.
   - New expander: "Classify new image" — upload an image, compute its spectrum, display rank profile and per-class scores.

3. **Evaluation** — replace or complement `logreg_cv`:
   - Leave-one-out or k-fold: for each held-out image, build atlas from remaining images, score against all classes, check if argmax is correct.
   - Compare accuracy to current logistic regression baseline.

## Open Questions

- **Band limiting**: should the percentile atlas respect the `[fmin, fmax]` band slider, or use the full spectrum? Probably yes — the band limits are physically motivated.
- **Weighting bins**: high-frequency bins have more variance and less class-discriminative signal. A frequency-weighted score (weight by between-class variance / within-class variance, i.e., a 1D Fisher criterion per bin) could sharpen the classifier.
- **Atlas size**: `n_classes × n_bins × 100` floats. For 4 classes, 200 bins, float32: 320 KB. Negligible.

---

## Ideas for Extension

### Crop size reduction

Since only bins 1–250 carry discriminative signal, the crop used to build each fingerprint can be made much smaller than the full image. A crop of side length `s` produces radial bins up to `s/2`; to reach bin 250 we only need `s ≥ 500 px`. Smaller crops mean:

a) **More training samples** — a single full image can yield multiple non-overlapping crops, multiplying the atlas training set with no additional labelled images.

b) **Per-pixel classification** — a crop of the minimum required size can be centered on any pixel. Each pixel gets its own fingerprint by running the Hann-windowed FFT on the crop centered there, then scoring against the atlas. This is the natural extension of the current "center crop of the full image" approach to a spatially dense version.

### Classification map via grid sampling + interpolation

Run per-pixel classification on a sparse grid (e.g. every 32 or 64 px) across the full image. Interpolate the per-class scores (or the winning class index) at all intermediate pixels using biquadratic (or bicubic) interpolation. The grid density is a direct performance knob: coarser grid → faster, finer grid → sharper boundaries. The interpolated score maps can be visualised as smooth heatmaps before thresholding to a hard class assignment.

### Colour overlay

Apply a per-class colour tint (e.g. the existing `CLASS_COLORS` palette) to the original grayscale image based on the classification map. Alpha-blend the tint over the raw image so texture remains visible underneath. This gives an immediate spatial reading of grinding quality across the surface.

### Outlier / "hole" class

A 5th atlas class for outliers (holes in the part, strong specular reflections, sensor artifacts). Two ways to populate it:

- **Manual labelling** — crop regions visually identified as outliers and add them to the atlas.
- **Spectral outlier mining** — images already flagged by the Mahalanobis filter have atypical spectra within their nominal class; collect these as a candidate outlier set and inspect them. From what has been observed, the dominant outlier types are holes in the part under inspection and unusually shaped specular reflections.

The MAD scorer naturally handles a 5-class atlas with no code changes — the outlier class simply competes for the argmax.

### GPU shader implementation

For production-speed classification maps, implement the per-pixel FFT + atlas scoring as a GPU compute or fragment shader:

- **Input textures**: the grayscale image + the atlas as a `(n_classes, n_bins, 100)` 3D texture.
- **Per-fragment work**: for each output pixel, sample the image in a `s × s` neighbourhood, apply a Hann window (precomputed as a 1D texture → outer product in the shader), compute the 2D FFT (e.g. via a two-pass row/column FFT using shared memory in a compute shader), accumulate radial bins, score against the atlas, emit the winning class or a blended colour.
- **Framework options**: GLSL compute shaders (OpenGL/Vulkan), WGSL (WebGPU, browser-deployable), or CUDA/ROCm for offline batch processing. WebGPU is attractive for a browser-based Streamlit alternative.
- **FFT in a shader**: a radix-2 Cooley–Tukey FFT over `s = 512` takes ~9 passes of butterfly operations on the GPU; well within real-time budget per tile. Alternatively, accumulate the radial profile directly in the spatial domain using the Wiener–Khinchin theorem (autocorrelation → power spectrum), which maps cleanly to a convolution pass.
