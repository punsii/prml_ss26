# Reflection Removal for Metal Surface Classification

## Problem

Strong light reflections on metal surfaces interfere with a classification model that identifies grinding states:

1. **Initial state** — unprocessed surface
2. **Grinding started** — early grinding marks visible
3. **Grinding decent** — significant material removal
4. **Polished/finished** — final surface quality

Reflections from workshop lighting create bright spots and specular highlights that obscure the actual surface texture, degrading classifier accuracy.

## Goal

Evaluate and implement reflection removal / brightness normalization techniques as a preprocessing step to improve grinding state classification.

## Implemented Methods

All scripts live in `src/` and can be run standalone or via Nix apps on the BMW test image.

### CLAHE — `src/clahe.py`

Contrast Limited Adaptive Histogram Equalization. Divides image into tiles, equalizes histogram per tile with contrast limiting. Applied to L channel in LAB space.

```
nix run .#runClaheBmw
python src/clahe.py <image> --clip-limit 2.0 --tile-size 8
```

### Multi-Scale Retinex — `src/retinex.py`

Decomposes image into illumination × reflectance in log domain. Subtracts Gaussian-blurred illumination at multiple scales to normalize local brightness.

- Ref: Jobson et al. (1997) "A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes." IEEE TIP 6(7).

```
nix run .#runRetinexBmw
python src/retinex.py <image> --sigmas 15 80 250
```

### Homomorphic Filtering — `src/homomorphic.py`

High-pass filter in log-frequency domain. Attenuates slowly-varying illumination, preserves surface texture.

- Ref: Gonzalez & Woods (2008) "Digital Image Processing" Ch. 4.9.

```
nix run .#runHomomorphicBmw
python src/homomorphic.py <image> --gamma-low 0.3 --gamma-high 1.5 --cutoff 30
```

### Specular/Diffuse Separation — `src/specular_diffuse.py`

Chromaticity-based separation using the dichromatic reflection model. Best when material and illuminant colors differ.

- Ref: Tan & Ikeuchi (2005) "Separating Reflection Components of Textured Surfaces using a Single Image." IEEE TPAMI 27(2).

```
nix run .#runSpecularBmw
python src/specular_diffuse.py <image> --iterations 1 --kernel-size 15
```

## Future Options

- **Highlight inpainting** — Mask saturated pixels, inpaint from surrounding texture.
- **Deep Learning (GANs)** — GCNet-based removal from [reference repo](https://github.com/Devashi-Choudhary/Reflection-Removal-Techniques-Review). Needs pretrained weights.
- **Convex Optimization** — Gradient thresholding + DCT (Yang et al. CVPR 2019). [MATLAB ref](https://github.com/alexch1/ImageProcessing).
- **Relative Smoothness** — Single image layer separation. [MATLAB ref](https://github.com/yyhz76/reflectSuppress).

## Dataset

Custom dataset of metal surfaces under various lighting conditions — to be added.
