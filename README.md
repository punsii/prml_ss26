# Reflection Removal for Metal Surface Classification

Strong specular reflections on metal workpieces interfere with a classifier that identifies four grinding states — *initial*, *grinding started*, *grinding decent*, *polished*. This repo evaluates reflection-removal / brightness-normalization techniques as a preprocessing step and explores frequency-domain features for the classifier itself.

## What's in here

Preprocessing methods (`src/`):

- `clahe.py` — Contrast Limited Adaptive Histogram Equalization
- `retinex.py` — Multi-Scale Retinex
- `homomorphic.py` — Homomorphic filtering (log-frequency high-pass)
- `specular_diffuse.py` — Dichromatic specular/diffuse separation

Classifier experiments (`src/`):

- `radial.py`, `class_stats.py` — radial FFT spectrum features
- `percentile_model.py` — per-class percentile-model classifier ("model" = atlas of per-wavelength percentiles, the analogue of weights in deep-learning)

UI:

- `app.py` — Streamlit app comparing methods side-by-side on the BMW dataset

## Running

Nix-based; no `pip` / `conda` needed. From the repo root:

```
nix develop                 # dev shell with python + deps
nix run .#runStreamlit      # launch the comparison UI
nix run .#runClaheBmw       # one-shot demo on a BMW reference image
nix run .#runRetinexBmw     # (also: runHomomorphicBmw, runSpecularBmw)
```

Each script in `src/` is also a standalone CLI — pass `--help` for options.

## Deployment

A NixOS module is exposed at `nixosModules.reflectionRemoval` that runs the streamlit app as a systemd service behind a Caddy reverse proxy. Hostname, ACME cert, data directory and flake URL are module options.

## See also

- [`docs/next_steps.md`](docs/next_steps.md) — current priorities and decided approach for upcoming work
- [`docs/research_progress.md`](docs/research_progress.md) — running notes on experiments and ideas
- [`docs/fft_spectrum_classifier.md`](docs/fft_spectrum_classifier.md) — percentile-model classifier design
- [`docs/rbfnn_research.md`](docs/rbfnn_research.md) — RBF network classifier notes
- [`docs/presentation_notes.md`](docs/presentation_notes.md) — slide outline and chronological notes
