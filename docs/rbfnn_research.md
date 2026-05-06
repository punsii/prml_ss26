# RBFNN for Metal Surface Grinding-State Classification

## What is an RBFNN

A Radial Basis Function Neural Network is a two-layer architecture: a hidden layer of RBF units (typically Gaussian), each computing a distance-based activation from a learned center, followed by a linear output layer. The hidden layer projects the input into a high-dimensional feature space where classes become linearly separable. Training is often split: centers are placed via k-means or random sampling on training data, and only the output weights are fit by linear regression or gradient descent — making it fast compared to full backprop MLPs.

## Why It May Suit This Task

Radial frequency spectra are real-valued, moderate-dimensional (~50–200 bins), and encode rotation-averaged surface texture. Grinding states tend to differ in characteristic frequency bands (fine vs. coarse grind produces distinct spatial frequency signatures), and those differences are often local and nonlinear in feature space. RBF kernels are well-suited to such "blob-like" class structure because each unit fires strongly near a prototype and decays smoothly — naturally capturing per-class spectral prototypes. With 4 classes and likely hundreds of samples per class, the dataset is large enough to place meaningful centers but small enough that a simple, fast method is preferable over deep nets.

## Key Hyperparameters

| Parameter | Role | Tuning guidance |
|---|---|---|
| `n_components` (centers) | Controls expressivity | Start at 100–500; tune via CV on validation accuracy |
| `gamma` (RBF width) | Controls locality of each kernel | Grid-search over `[0.001, 0.01, 0.1, 1.0]`; too small → underfitting, too large → overfitting |
| Output regularization `C` | Regularizes the linear head | Same grid as logistic regression baseline |
| Center placement | k-means vs. random | k-means clusters often better than pure random for structured spectra |

Use `StratifiedKFold` (same as current baseline) for all tuning to preserve class balance.

## Comparison to Baseline and Alternatives

**Logistic Regression (baseline):** Linear decision boundaries in feature space. Will fail when class boundaries in spectrum-space are curved or when spectral modes overlap nonlinearly. RBFNN adds nonlinearity at low cost.

**SVM-RBF (`sklearn.svm.SVC(kernel='rbf')`):** Mathematically equivalent kernel machine with a principled margin objective. Generally more robust than RBFNN for small-to-medium datasets and avoids the center-placement heuristic. Strong first choice for a nonlinear upgrade from LR.

**Shallow MLP (`sklearn.neural_network.MLPClassifier`):** More flexible but requires more data and careful regularization. Harder to tune; not necessarily better than SVM-RBF at this scale.

**RBFNN vs. SVM-RBF:** For this task, SVM-RBF is likely the safer bet. RBFNN shines when inference speed on new samples matters (explicit feature map is precomputed) or when you need calibrated probabilistic outputs.

## Scikit-learn Implementation Sketch

```python
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rbf_features", RBFSampler(gamma=0.1, n_components=300, random_state=42)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial")),
])

param_grid = {
    "rbf_features__gamma": [0.01, 0.1, 1.0],
    "rbf_features__n_components": [100, 300, 500],
    "clf__C": [0.1, 1.0, 10.0],
}

search = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(5), scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)
```

`RBFSampler` uses random Fourier features to approximate the RBF kernel map — this is the standard scikit-learn approximation of a full RBFNN and is fast enough for real-time inference.

## Pitfalls and When to Prefer Something Else

- **Center placement matters:** Random centers (as in `RBFSampler`) may miss important spectral modes. If accuracy plateaus, replace with k-means centers using a custom transformer.
- **Gamma sensitivity:** Performance degrades sharply with wrong `gamma`; always cross-validate.
- **Scaling is mandatory:** RBF distances are meaningless without standardized inputs — `StandardScaler` must precede the kernel step.
- **Prefer SVM-RBF if:** dataset is small (<200 samples/class), you want a margin guarantee, or tuning budget is limited.
- **Prefer MLP if:** you add more feature types later (e.g., spatial texture + spectrum), as MLP handles heterogeneous inputs better.
- **Avoid RBFNN if:** interpretability of individual frequency bins is needed — the kernel projection destroys direct feature attribution.
