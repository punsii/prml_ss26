# Next Steps and Priorities

Snapshot taken 2026-05-07 after the professor's review feedback. Items are listed in dependency order — earlier items are prerequisites for later ones. Update as items land or priorities shift.

## 1. Wavelength-based radial axis (prerequisite for everything spatial)

**Status:** implemented (2026-05-07). See `radial.WAVELENGTHS`, `radial.MIN_IMAGE_SIZE`, `radial.radial_profile_at_wavelengths`. Class-spectra tab and percentile model already use the wavelength axis end-to-end.

**Motivation:** raw radial bins are not patch-size invariant. For an `N × N` patch, bin `r` corresponds to wavelength `N / r` pixels — so the same bin index means a different physical frequency on patches of different sizes. This blocks segmentation (a model built on full images cannot classify smaller patches) and is the root cause of the Labor dataset breaking under the current `FMAX=250` raw-bin limit.

**Decisions:**

- Wavelength axis: integer wavelengths from **2 px (Nyquist) to 200 px**, linear spacing. Log spacing rejected for now — revisit if data sparsity at high wavelengths becomes a problem.
- No NaN handling: images with `min(h, w) < 200 px` are dropped — wavelength 200 needs at least one full cycle in the patch.
- Plot orientation: short wavelengths on the **left**, long wavelengths on the right. This is reversed relative to the current bin-indexed plots, but it lets the axis extend rightward for longer wavelengths later without rescaling.
- Model portability: with a fixed wavelength axis, one model can score any patch size that covers the full range. Building models at multiple patch sizes is unnecessary as long as wavelengths line up.

**Implementation pointers:** add `radial_profile_at_wavelengths(image, wavelengths)` to `src/radial.py`. Refactor every raw-bin call-site to use the wavelength axis: `compute_full_image_vectors`, `_load_vectors_with_progress`, FFT explorer, percentile-model heatmap, waterfall, 3D point cloud. Drop `FMIN` / `FMAX` constants in favour of the wavelength range.

## 2. Image segmentation via patch raster + biquadratic interpolation

**Status:** documented in `fft_spectrum_classifier.md` ("Ideas for Extension"); requires step 1.

**Approach:**

- Patch size: **500 px** as the starting point. Covers wavelengths 2..500 px with comfortable margin around the discriminative band.
- Grid spacing: **100 px** starting point (~1300 patches per 3648-px image). Adjust based on observed wall-clock per image.
- Per-patch normalisation: median-subtract or z-score before FFT. Exact choice from a quick experiment.
- Edge handling: skip grid points that cannot centre a full-size patch.
- Interpolation target: per-class softmax scores (a 4-vector per grid point), **not** class index. Argmax (or score-weighted mean — see open question below) happens after interpolation, at every pixel.
- Visualisation: per-class colour tint over the original grayscale.

**Open question — gradient overlay:** map classes to `[0, ⅓, ⅔, 1]`, take the dot product with softmax probabilities → scalar per patch → continuous colour map. Cheap to add as a parallel visualisation alongside hard argmax. Caveat: a patch genuinely confused between class 0 and class 2 averages to ~class 1, which is wrong; tolerable for visualisation, not for decisions. The outlier class (step 3) is expected to absorb most actual "no class fits" cases.

## 3. Outlier 5th class via auto-mining + confirm/deny

**Status:** documented in `fft_spectrum_classifier.md` ("Ideas for Extension"); requires step 2.

**Approach:**

- After the patch classifier runs, rank patches by `max(score)` across the four classes. Low max → fits no class well → outlier candidate.
- Streamlit panel: bottom-K candidates as thumbnails + per-class score bars + 👍/👎 buttons. User confirms or denies each. Confirmed outliers go into a 5th-class training set.
- Build a 5th-class model from the confirmed set; re-classify; tint the 5th class in the segmentation map (default black; consider diagonal hatch or desaturated grey since pure black reads as "missing data").
- Manual cold-start labelling explicitly rejected as too much work — auto-mining + confirm/deny replaces it.

**Heterogeneity caveat:** the outlier set mixes holes, stamps, and specular artefacts. The percentile model may not have coherent per-wavelength distributions for it. Accept this as a starting point; the segmentation results will tell you whether sub-clustering (step 4) is needed.

## 4. Iterative refinement and outlier sub-clustering (future)

**Status:** future. Only relevant after 1–3 land and are evaluated.

- Re-running the classifier after the 5th class is added produces a new outlier set. Iterating *may* help, but needs a stopping rule (fixed iteration count, or: outlier set changes by <X% between rounds) to avoid eating into legitimate classes.
- Sub-cluster the confirmed outlier set into multiple classes (holes vs stamps vs reflections) using **HDBSCAN** on the spectrum vectors — handles non-convex clusters, naturally produces a "noise" label. K-means rejected because spectra rarely sit in convex Gaussian blobs.

## 5. Duplicate-image audit

**Status:** todo. Independent of 1–4; can be done in parallel.

Both datasets are suspected of containing duplicate or near-duplicate images, which would inflate cross-validation accuracy without breaking method-vs-method comparisons. Write a script that:

- Detects exact duplicates (e.g. SHA-256 of decoded pixels, plus perceptual hash to also catch losslessly-resaved JPEGs).
- Optionally detects near-duplicates (perceptual hash with a similarity threshold; normalised cross-correlation on downscaled grayscale).
- Lists candidate duplicate groups so the user can confirm and decide whether to deduplicate before retraining.

Details to be designed when the script is actually written. Worth running before any "ship/no ship" judgement on absolute classifier accuracy, but not blocking the wavelength refactor or segmentation.

## 6. Open questions / parking lot

- **Auto-scaling across cameras / standoff distances.** Discussed in `fft_spectrum_classifier.md` ("Scale Matching"). For now both datasets are assumed to share pixel pitch and lens. When that assumption breaks, the wavelength axis becomes "wavelength in mm" via per-image pixel-pitch metadata; the model structure is unchanged.
- **MAD-from-50 currently underperforms the logistic-regression baseline.** Likely cause: uneven per-wavelength variance (LR sees `StandardScaler`-normalised features; MAD scores raw log-magnitudes). Per-bin Fisher weighting (in `fft_spectrum_classifier.md` open questions) is the most likely remedy. Defer until step 1 lands — comparing MAD vs LR on raw-bin features now is the wrong moment.
