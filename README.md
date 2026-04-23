# Reflection Removal for Metal Surface Classification

## Problem

Strong light reflections on metal surfaces interfere with a classification model that identifies grinding states:

1. **Initial state** — unprocessed surface
2. **Grinding started** — early grinding marks visible
3. **Grinding decent** — significant material removal
4. **Polished/finished** — final surface quality

Reflections from workshop lighting create bright spots and specular highlights that obscure the actual surface texture, degrading classifier accuracy.

## Goal

Evaluate and implement reflection removal techniques as a preprocessing step to improve grinding state classification. Starting point is [Reflection-Removal-Techniques-Review](https://github.com/Devashi-Choudhary/Reflection-Removal-Techniques-Review).

## Approach

1. Implement/adapt reflection removal methods from the reference repo
2. Test on our metal surface dataset (to be added)
3. Measure impact on classification accuracy with and without reflection removal
4. Select best method for integration into the classification pipeline

## Immediate Goals

1. **Specular/diffuse separation** — Implement chromaticity-based separation (Tan & Ikeuchi 2005, Yang et al. 2010). Separates specular highlights from diffuse component. Well-suited for metal since specular highlights shift toward light source color.

## Future Options

- **Highlight inpainting** — Detect saturated/near-saturated pixels, mask them, inpaint from surrounding texture. Simple and effective when highlights are small relative to surface area.
- **Deep Learning (GANs)** — GCNet-based reflection removal from [reference repo](https://github.com/Devashi-Choudhary/Reflection-Removal-Techniques-Review). Single-image, needs pretrained weights.
- **Convex Optimization** — Reflection suppression via gradient thresholding + DCT (Yang et al. CVPR 2019). [MATLAB reference](https://github.com/alexch1/ImageProcessing).
- **Relative Smoothness** — Single image layer separation. [MATLAB reference](https://github.com/yyhz76/reflectSuppress).

## Dataset

Custom dataset of metal surfaces under various lighting conditions — to be added.
