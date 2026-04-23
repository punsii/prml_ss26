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

## Dataset

Custom dataset of metal surfaces under various lighting conditions — to be added.
