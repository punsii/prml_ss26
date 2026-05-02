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
