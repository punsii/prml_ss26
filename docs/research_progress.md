# Research Progress

A chronological account of the ideas explored, experiments run, and decisions taken on the way from "remove specular reflections from metal images" to "classify grinding state via radial FFT spectra."

## Problem framing

A machine grinds and polishes a metal part through four progressively finer stages, and a vision system grades each stage from images of the surface. Specular highlights from the lighting rig were assumed to dominate the noise budget, so the original goal was reflection removal as preprocessing. The project gradually reframed itself: spectral features turned out to be discriminative *with* the highlights present, and the preprocessing methods became interchangeable front-ends to a feature extractor rather than the central object of study.

Two datasets back the work:

- **Datensatz Labor** — CSV-labeled lab images.
- **BMW 25** — production images, four directory-named wear classes (`Step_0_17-25ym`, `Step_1_10-17ym`, `Step_2_3-10ym`, `Step_3_final_3ym`). Image resolution 3648×3648 px; grinding-line features at the ~50 px scale.

## Phase 1 — Literature scoping (2026-04-23)

Off-the-shelf reflection-removal methods evaluated: **image averaging across exposures** and **ICA-based separation** (`067d8b2`). Both wired up as Nix derivations with `feh` viewers.

**Decision:** neither integrated. Averaging needs registered multi-shot data we don't have; ICA is brittle on single images of textured matte/specular composites. The first in-house method, **specular/diffuse separation** based on the dichromatic reflection model (`specular_diffuse.py`, `7178ae8`), survived this phase.

## Phase 2 — Brightness-normalization toolbox (2026-04-23 → 2026-04-30)

Pivot from "two-layer scene decomposition" to "illumination/contrast normalization." Three classical methods added together (`c3cf1bb`):

- **CLAHE** (`src/clahe.py`) — tile-based histogram equalization with contrast clipping. Doesn't physically separate illumination — clipped pixels stay clipped.
- **Multi-Scale Retinex** (`src/retinex.py`) — log-domain `image / blur(image)` at three scales (σ ∈ {15, 80, 250}) per Jobson et al. 1997. Tends to wash out the surface.
- **Homomorphic filtering** (`src/homomorphic.py`) — Gaussian high-pass on `log(image)` in the FFT domain. Tends dark.

Two architectural decisions from this phase shaped everything afterwards:

- **Patch mode (`de799f9`)** — split the 3648×3648 image into a 9×9 grid; process only the 9 inner tiles at indices `(1, 4, 7)`. Outer tiles act as a buffer so kernels never need to pad from outside the image. ~11× cheaper FFT, no edge artifacts, locally-homogeneous statistics per tile.
- **Grayscale switch (`6c71c70`)** — explicit reasoning: "metal is monochromatic; the signal is texture, not color." 3× cheaper FFT, no LAB round-trip. Silent consequence: specular/diffuse separation depends on chromaticity and was implicitly retired here.

## Phase 3 — Pivot to spectral classification (2026-04-30)

Three commits on the same day moved the project's centre of gravity from "make the image cleaner" to "extract a feature vector":

- **`82d8bfa` Radial spectrum analysis** — replaced 2D FFT thumbnails with 1D radial profiles. Two FFT artefacts had to be fixed first:
  - *Boundary cross* in the 2D spectrum from periodic wrap-around → fixed by **Hann windowing** the input before the FFT.
  - *Anisotropic radial sampling* — radii beyond `r = N/2` only see corner samples → fixed by **cropping the radial profile to the inscribed circle**.

  Direction is irrelevant for grading (only magnitude per period matters), so radial averaging collapses 2D → 1D without information loss for our purposes.

- **`7d916e6`** extracted helpers into `src/radial.py`.

- **`17fd286` `class_stats` library + Plotly tab** — operationalized the next step. New module exposes `compute_full_image_vectors`, `compute_class_stats`, `mahalanobis_filter`, and the first classifier baseline `logreg_cv` (StandardScaler → multinomial logistic regression → 5-fold StratifiedKFold). Class-spectra tab moved to Plotly with PCA 3D scatter and per-class radial spectra.

From this point the radial spectrum is the project's central object. The four preprocessing methods (raw, CLAHE, Retinex, homomorphic) are interchangeable front-ends to it.

## Phase 4 — Classifier exploration and dataset expansion (2026-05-01 → 2026-05-02)

- **`6534fe8` BMW dataset, FFT explorer, frequency band-limiting** — second dataset wired in. Added `[fmin, fmax]` band slider so the user can drop low bins (DC + uniform brightness) and high bins (sensor noise) and isolate the band where grinding lines actually live.

- **`93adfbc` Parallel spectra loading** — `ThreadPoolExecutor(max_workers=8)` for the per-image FFT pass; performance only.

- **`f58182f` RBFNN feasibility notes** (`rbfnn_research.md`) — three nonlinear upgrades over the LR baseline considered: RBFNN (`RBFSampler` + `LogisticRegression`), SVM-RBF flagged as "the safer bet" at this dataset scale, MLP only if heterogeneous features are added later. **No code was written** — interpretability of individual frequency bins was prioritised, which the kernel projection destroys.

- **`0fa3614` / `849ffb3` Waterfall plot** — 3D per-image spectra surface in the class-spectra tab.

- **`7365577` Percentile-atlas design doc** (`fft_spectrum_classifier.md`):

  - Build a `(n_classes, n_bins, 100)` tensor: per class, per bin, the empirical 1–100 percentile thresholds across training spectra.
  - Score a query by ranking each bin value within the class atlas (rank ∈ [0, 100]). A spectrum that fits class `c` clusters its ranks near 50 (median).
  - Four candidate scoring functions: **MAD from 50** (`-mean(|R - 50|)`), Gaussian log-likelihood, rank-histogram entropy, Mahalanobis distance.
  - **Scale matching is required first** — different camera resolutions map the same physical frequency to different bins. Nyquist normalisation (`bin / n_bins`) is the simplest fix.
  - Percentile and Mahalanobis converge under Gaussian distributions; percentiles win on heavy-tailed or multimodal bins.

## Current state

- **`src/percentile_atlas.py`** — implements the design doc with **MAD-from-50** as the chosen first metric. Exposes `build_atlas`, `score_spectrum`, `rank_profiles`, `predict`, `mad_loo_cv` (leave-one-out CV).
- **Class-spectra tab additions** in `app.py`:
  - "Percentile atlas (per-class)" expander — heatmaps of the percentile tensor (bins × percentiles).
  - "Classification accuracy (filtered)" comparison table — LR 5-fold next to MAD LOO on the same filtered vectors.
  - "Classify new image" expander — upload, score, predicted class, score bar chart, per-bin rank profiles with a dashed median line.
- Mahalanobis-outlier thumbnail strip redesigned: larger images (280 px), processed column hidden when method is `raw`, FFT magnitude image added as a column.

## Open questions and future work

Documented in `fft_spectrum_classifier.md`:

- **Score function choice** — MAD-from-50 picked as the simplest starting point. Once a benchmark exists, the other three (Gaussian log-likelihood, rank-histogram entropy, Mahalanobis) can be swapped in with no other code changes.
- **Per-bin Fisher weighting** — weight bins by between-class / within-class variance to suppress non-discriminative high-frequency bins.
- **Crop-size reduction → per-pixel classification map** — only bins 1–250 are discriminative, so a ~500 px crop suffices. That unlocks:
  - (a) more training fingerprints per image,
  - (b) running the classifier centred on any pixel,
  - (c) sparse-grid sampling + biquadratic interpolation for a full image-wide classification map; grid density as a performance knob,
  - (d) class-coloured tint overlay on the original grayscale,
  - (e) a 5th "outlier" class trained on holes / weird specular reflections (either manually labelled or harvested from the existing Mahalanobis-flagged set),
  - (f) GPU shader implementation using a `(n_classes, n_bins, 100)` 3D atlas texture for production-speed classification maps.

## Narrative summary

- **The decisive pivot is `82d8bfa` + `17fd286` on 2026-04-30.** Reflection removal stopped being the goal and became one of four interchangeable preprocessing options in front of the radial-FFT feature extractor.
- **Nothing was deleted by an explicit "remove" commit.** Specular/diffuse separation was implicitly retired at the grayscale switch; the external averaging/ICA experiment never returned after Phase 1.
- **ML approaches in chronological order:** logistic regression CV baseline → Mahalanobis outlier filter → RBFNN/SVM-RBF (notes only) → percentile atlas with MAD-from-50 (current frontier).
