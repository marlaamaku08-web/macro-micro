# PathMNIST Model Comparison

_Generated: 2026-10-15T12:58:03+00:00 — `train_compare_pathmnist.py`, seed 42_

## Summary (untouched PathMNIST test set, n = 7,180)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Model A — CNN | 87.06% | 83.61% | 82.80% | 81.88% |
| Model B — PCA + LogisticRegression | 62.63% | 57.13% | 53.98% | 52.05% |

**Winner: Model A — CNN** — selected by macro_f1 (tie-break: accuracy). Macro F1 difference: 29.83 percentage points; accuracy difference: 24.43 percentage points.

Model confidence / predicted probability is **not** used as a comparison metric.

## Task

Both models solve the same task: 9-class colorectal histopathology tissue classification (PathMNIST). The macro/micro SVM model in this project solves a different task and is **not** part of this comparison.

## Dataset and splits

- Dataset file: `pathmnist_64.npz` (SHA-256 `1e7fc200dd5aac79…`)
- Image size: [64, 64, 3] (RGB, uint8)
- Official train split: 89,996 images
- Official validation split: 10,004 images (Model B: reported only, not used for fitting or tuning)
- Official test split: 7,180 images — used **only** for final evaluation, identical for both models, not subsampled

### Data leakage check

- Split arrays are separate keys in the `.npz` file; the test array is loaded only after both models are trained/loaded.
- Exact-duplicate images (SHA-1 of pixel bytes) between Model B training subset and test set: **0**
- Exact-duplicate images between the full train split and test set: **0**
- Hyperparameters of Model B are fixed in code (no search on the test set).

## Model B training subset

- Model B is trained on a **stratified random subset of 25,000** of the 89,996 training images (seed 42), to keep RAM and training time manageable.
- Model A (CNN) was trained earlier by `train_pathmnist.py` on the full training split.
- Class distribution of the subset matches the full training split (stratified).

| Class | Full train | Model B subset |
|---|---|---|
| 0 Adipose | 9,366 | 2,602 |
| 1 Background | 9,509 | 2,642 |
| 2 Debris | 10,360 | 2,878 |
| 3 Lymphocytes | 10,401 | 2,889 |
| 4 Mucus | 8,006 | 2,224 |
| 5 Smooth muscle | 12,182 | 3,384 |
| 6 Normal colon mucosa | 7,886 | 2,191 |
| 7 Cancer-associated stroma | 9,401 | 2,611 |
| 8 Colorectal adenocarcinoma epithelium | 12,885 | 3,579 |

## Preprocessing

- Both models: pixel values scaled to [0, 1] by dividing by 255 (same as `app.py` inference).
- Model A: input tensor 64×64×3.
- Model B: image flattened to 12,288 features → StandardScaler → PCA.

## Model architectures

**Model A — CNN** (`pathmnist_model.keras`, loaded, not retrained)

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 62, 62, 32)     │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 31, 31, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 29, 29, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 14, 14, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 12, 12, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 6, 6, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 4608)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │       589,952 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 9)              │         1,161 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2,053,085 (7.83 MB)
 Trainable params: 684,361 (2.61 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 1,368,724 (5.22 MB)

```

**Model B — PCA + LogisticRegression** (`pathmnist_logreg.pkl`)

```
Pipeline:
  1. StandardScaler()
  2. PCA(n_components=100, svd_solver='randomized', random_state=42)
  3. LogisticRegression(C=1.0, max_iter=1000, random_state=42)  # multinomial (softmax) for 9 classes
Input: 64x64x3 image flattened to 12,288 features
```

Memory/time optimizations for Model B: stratified training subset, float32 features, randomized SVD PCA (100 components), batched test-set inference.
Model B training time: 9.5 s; PCA explained variance: 68.94%; validation accuracy (info only): 52.56%.

## Per-class metrics (test set)

| Class | Support | A Precision | A Recall | A F1 | B Precision | B Recall | B F1 |
|---|---|---|---|---|---|---|---|
| 0 Adipose | 1,338 | 0.9191 | 0.9851 | 0.9509 | 0.9282 | 0.8692 | 0.8977 |
| 1 Background | 847 | 0.9217 | 1.0000 | 0.9592 | 0.7663 | 0.9988 | 0.8672 |
| 2 Debris | 339 | 0.4811 | 0.8643 | 0.6181 | 0.1874 | 0.2360 | 0.2089 |
| 3 Lymphocytes | 634 | 0.9455 | 0.9858 | 0.9653 | 0.5051 | 0.6972 | 0.5858 |
| 4 Mucus | 1,035 | 0.9738 | 0.8271 | 0.8945 | 0.6881 | 0.6502 | 0.6687 |
| 5 Smooth muscle | 592 | 0.7495 | 0.6216 | 0.6796 | 0.4143 | 0.4409 | 0.4272 |
| 6 Normal colon mucosa | 741 | 0.9235 | 0.8475 | 0.8839 | 0.5982 | 0.1808 | 0.2777 |
| 7 Cancer-associated stroma | 421 | 0.7074 | 0.3848 | 0.4985 | 0.5217 | 0.0855 | 0.1469 |
| 8 Colorectal adenocarcinoma epithelium | 1,233 | 0.9030 | 0.9359 | 0.9192 | 0.5321 | 0.6991 | 0.6043 |

## Confusion matrices

Rows = true label, columns = predicted label (counts and row-normalized %).

- Model A: [`confusion_matrix_cnn.png`](confusion_matrix_cnn.png)
- Model B: [`confusion_matrix_logreg.png`](confusion_matrix_logreg.png)

## Limitations

- Model B uses a training subset while Model A used the full training split, so the comparison also reflects training-data volume, not only model family.
- Model A was trained earlier without fixed seeds; its evaluation here is deterministic, but retraining it would give slightly different weights.
- PathMNIST train/validation patches come from NCT-CRC-HE-100K and the test patches from CRC-VAL-HE-7K (a separate patient cohort), all colorectal H&E at 64×64; performance on other organs, stains, scanners or magnifications is unknown.
- Metrics describe patch-level tissue classification for education/research, not diagnosis.
- Softmax / predicted probability is not a measure of clinical certainty.

## Environment

- Python 3.12.15, numpy 2.5.4, scikit-learn 1.9.1, tensorflow 2.21.0
