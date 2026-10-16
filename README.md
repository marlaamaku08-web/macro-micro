# Macro&Micro

_AI-Powered Pathology Learning_

**AI-assisted pathology learning prototype — for education and research only. Not a clinical diagnostic system.**

---

## 1. Project Overview

> **Naming:** the product's display name is **Macro&Micro**. Technical identifiers created earlier
> keep the original `macro2micro` spelling (for example `macro2micro_model.pkl`,
> `db/macro2micro.db`, `MACRO2MICRO_MODEL_PATH`) so existing paths and configuration keep working.

Macro&Micro is a web-based **educational and research prototype** that helps learners connect
macroscopic (gross) pathology with microscopic (histology) findings.

A user uploads an image and the system:

1. classifies the image type — **macroscopic vs. microscopic** (small classical ML model);
2. for PathMNIST patches, classifies the **colorectal tissue type** (9 classes, CNN);
3. shows educational content — morphology, explanation, pathogenesis and, for ToxoPath-1
   cases, the related gross image.

The outputs are learning aids. They are **not** diagnoses and must not be used for patient care.

## 2. Problem and Users

**Users:** medical and pathology students, and instructors preparing teaching material.

**Problem:** linking what is seen at gross level with what is seen under the microscope, and
recognizing basic histopathology tissue types, takes a lot of guided practice. Macro&Micro
provides an interactive way to practice: upload an image, see what the model predicts, and read
the morphology and explanation for that class.

## 3. System Architecture

| # | Layer | Responsibility | Implementation |
|---|---|---|---|
| 1 | **User Interface** | Upload form, image preview, result cards, confidence bars, Pathology Learning content, reset | [`index.html`](index.html) |
| 2 | **Service / API** | Flask routes, `POST` endpoints, input validation, JSON responses, error codes | [`app.py`](app.py), [`services/validation.py`](services/validation.py) |
| 3 | **Business Logic** | Model output → decision (`classified` / `low_confidence`), confidence threshold, labels, educational explanations, dataset detection | [`services/business_logic.py`](services/business_logic.py) |
| 4 | **Database** | SQLite logging of analyses and predictions (2 tables, FOREIGN KEY, parameterized SQL) | [`services/database.py`](services/database.py), [`db/schema.sql`](db/schema.sql) |
| 5 | **Model** | Load saved models once per process and run inference | [`services/model_loader.py`](services/model_loader.py), `macro2micro_model.pkl`, `pathmnist_model.keras`, `pathmnist_logreg.pkl` |

Configuration (paths, limits, threshold, debug flag) is read from environment variables in
[`config.py`](config.py); see [`.env.example`](.env.example).

```
Browser (index.html)
   │  multipart/form-data
   ▼
app.py ── validation.py ── model_loader.py ── business_logic.py ── database.py ── SQLite
   │                                                                     
   └── JSON response (class, label, confidence, decision, explanation, …)
```

## 4. Dataset

The project uses **two separate datasets for two different purposes**. They are not combined
and are not treated as one dataset.

### ToxoPath-1 (case-based educational set)

- A **very small**, curated set of gross and histology images of **cerebral and cardiac
  toxoplasmosis** (`images/Gross_images`, `images/Histology_brain`, `images/histology_heart`).
- Used for the case-based Pathology Learning workflow (organ, label, morphology, pathogenesis,
  related gross image) and to train the small macro/micro image-type model.
- **Limitation:** only about a dozen images (3 gross + 7 histology after cleanup). It is suitable
  for demonstration and teaching, not for training or validating a reliable classifier.

### PathMNIST (MedMNIST)

- **9-class colorectal histopathology** tissue classification, 64×64 RGB H&E patches
  (`pathmnist_64.npz`). Train/validation patches come from NCT-CRC-HE-100K; test patches come
  from CRC-VAL-HE-7K (a separate patient cohort).
- Official splits: **89,996 train / 10,004 validation / 7,180 test** images.
- Classes: 0 Adipose, 1 Background, 2 Debris, 3 Lymphocytes, 4 Mucus, 5 Smooth muscle,
  6 Normal colon mucosa, 7 Cancer-associated stroma, 8 Colorectal adenocarcinoma epithelium.
- Public, de-identified research dataset; no personal information is used as a feature.

## 5. Models

### A. Macro/Micro image-type model — `macro2micro_model.pkl`

- Pipeline: `StandardScaler → PCA (95% variance) → linear SVC`, trained by
  [`train_model.py`](train_model.py) on flattened 64×64 RGB pixels.
- Trained on the **tiny ToxoPath-1 image set** — an educational demonstration of a classical ML
  pipeline. Its test split is only a few images, so its accuracy is **not reliable** and it should
  not be used to judge arbitrary images.
- **Known label issue in the saved model:** the current `macro2micro_model.pkl` was trained with
  `images/Gross_images/Ukrainian_microscope_(cropped).jpg` labelled as *macro*, although it is a
  photograph of microscope equipment, not a pathology specimen. `train_model.py` now excludes this
  file (`EXCLUDED_FILES`), but the saved model has **not** been retrained, to keep the demo
  behaviour unchanged. Re-running `train_model.py` would train on 3 macro + 7 micro images and
  **overwrite** `macro2micro_model.pkl`.
- **Not** part of the Model A vs. Model B comparison (different task).

### B. PathMNIST Model A — CNN — `pathmnist_model.keras`

- 3 × (Conv2D + MaxPooling) → Dense(128) → Dropout(0.3) → Dense(9, softmax),
  trained by [`train_pathmnist.py`](train_pathmnist.py) on the full training split.
- This is the model used by the website (`POST /predict-pathology`).

### C. PathMNIST Model B — PCA + LogisticRegression — `pathmnist_logreg.pkl`

- Pipeline: `StandardScaler → PCA(100, randomized SVD) → LogisticRegression`, trained by
  [`train_compare_pathmnist.py`](train_compare_pathmnist.py) on a stratified, seeded subset of
  the training split (to keep RAM and training time manageable).
- Baseline model for the comparison.

**Model A and Model B solve the same task (PathMNIST 9-class classification) and are evaluated on
the same, untouched PathMNIST test set.**

## 6. Model Evaluation

Both PathMNIST models were evaluated by [`train_compare_pathmnist.py`](train_compare_pathmnist.py)
on the **official PathMNIST test split (7,180 images)**. The test split is never used for
training or for choosing hyperparameters, it is the same array for both models, and it is not
subsampled. Metrics are recomputed from the saved models every run (seed 42); two
consecutive runs produced identical metrics and identical predictions.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Model A — CNN | 87.06% | 83.61% | 82.80% | 81.88% |
| Model B — PCA + LogisticRegression | 62.63% | 57.13% | 53.98% | 52.05% |

**Result:** Model A — CNN performs better, selected by macro F1 (each class weighted equally;
accuracy as tie-break) — +29.83 percentage points macro F1 and
+24.43 percentage points accuracy. Predicted probability /
"confidence" is not used as a comparison metric.

- **Data leakage check:** 0 exact-duplicate images between the
  training split and the test split; the test array is loaded only after both models are ready.
- **Model B subset:** trained on a stratified, seeded subset of 25,000 of the
  89,996 training images (memory/time). Model A used the full training split, so the gap
  partly reflects training-data volume.
- **Hardest classes for both models:** Cancer-associated stroma (F1 A 0.50 / B 0.15)
  and Debris (F1 A 0.62 / B 0.21), which are often confused with
  each other and with Smooth muscle.

Full report with per-class precision/recall/F1/support, splits, preprocessing and architectures:
[`reports/model_comparison.md`](reports/model_comparison.md) ·
[`reports/model_comparison.json`](reports/model_comparison.json) · confusion matrices:
[`reports/confusion_matrix_cnn.png`](reports/confusion_matrix_cnn.png),
[`reports/confusion_matrix_logreg.png`](reports/confusion_matrix_logreg.png).

## 7. Database

SQLite database with two tables ([`db/schema.sql`](db/schema.sql)):

| Table | Columns | Purpose |
|---|---|---|
| `analyses` | `id` (PK), `created_at`, `dataset`, `file_ext` | One upload / analysis request |
| `predictions` | `id` (PK), `analysis_id` (FK), `model_name`, `predicted_class`, `label`, `confidence`, `decision` | Model output for that analysis |

- **Relationship:** `predictions.analysis_id` → `analyses.id` (`FOREIGN KEY … ON DELETE CASCADE`).
  `PRAGMA foreign_keys = ON` is set on every connection, so orphan predictions are rejected.
- **Parameterized SQL only:** every statement in [`services/database.py`](services/database.py)
  is a constant string with `?` placeholders; values are passed as a tuple. No f-strings or string
  concatenation are used to build SQL (checked by an automated AST test).
- **No PII and no images:** the app does not ask for the user's name or any other personal
  information. The original file name and the image bytes are
  **not stored**. Only dataset name, file extension and model outputs are logged.
- **No credentials:** SQLite is a local file database and does not use a username or password.
  The only setting is the file location, `DATABASE_PATH` (default `db/macro2micro.db`), read from
  the environment in [`config.py`](config.py). If a server database were used later, its
  credentials would also come from environment variables, never from source code.
- Logging failures are caught and logged; they never break the prediction response.

## 8. API

| Endpoint | Purpose | Model |
|---|---|---|
| `POST /predict` | Image type: macroscopic vs. microscopic | `macro2micro_model.pkl` |
| `POST /predict-pathology` | PathMNIST tissue class (9 classes) | `pathmnist_model.keras` |

**Request:** `multipart/form-data` with field `image`.

**Validation** ([`services/validation.py`](services/validation.py)):

| Case | Status |
|---|---|
| No file / empty filename | `400` |
| Extension not `.png`, `.jpg`, `.jpeg` | `415` |
| File is not a readable PNG/JPEG image | `400` |
| Larger than `MAX_UPLOAD_MB` (default 5 MB) | `413` |
| Model missing / unexpected server error | `500` |

**Response (200):**

```json
{
  "class": 8,
  "label": "Colorectal adenocarcinoma epithelium",
  "confidence": 99.98,
  "notice": "PathMNIST-д суурилсан сургалтын AI. Эмнэлзүйн оношилгоонд ашиглахгүй.",
  "decision": "classified",
  "decision_message": null,
  "explanation": "…",
  "morphology": "…",
  "dataset": "PathMNIST",
  "analysis_id": 4
}
```

`class`, `label`, `confidence` and `notice` are kept for backward compatibility with the frontend.
Errors return `{"error": "<message>"}`.

## 9. Business Logic

Implemented in [`services/business_logic.py`](services/business_logic.py):

- **Classification:** maps the raw model output (class index / class name) to a human-readable
  label.
- **Confidence threshold:** `CONFIDENCE_THRESHOLD` (default `0.60`, configurable).
  - probability ≥ threshold → `decision = "classified"`
  - probability < threshold → `decision = "low_confidence"` with the message
    *"Итгэлцлийн түвшин бага. Үр дүнг сургалтын хүрээнд дахин нягтална уу."*
- **Educational explanation:** morphology and explanation text for each of the 9 PathMNIST
  classes, and a short explanation for macro vs. micro images.
- **Dataset separation:** `detect_dataset(filename)` treats files starting with `pathmnist_` as
  PathMNIST and everything else as ToxoPath-1. The UI uses the same rule so PathMNIST results and
  ToxoPath-1 case content are never shown together.

## 10. Installation

Requires **Python 3.12** (tested with 3.12.15 on macOS).

```bash
python3.12 -m venv .venv-pathmnist
source .venv-pathmnist/bin/activate
pip install -r requirements.txt
```

Optional configuration:

```bash
cp .env.example .env        # edit values if needed
set -a; source .env; set +a
```

## 11. Run

```bash
python app.py
```

Open <http://127.0.0.1:5000>.

Other commands:

```bash
# Re-run the PathMNIST model comparison (writes reports/ and pathmnist_logreg.pkl)
python train_compare_pathmnist.py

# Automated regression tests (API, validation, database, SQL safety, frontend logic)
python -m unittest discover -s tests -t .
```

On macOS, port 5000 can be taken by *AirPlay Receiver*; set `PORT=5001` if needed.

## 12. Limitations and Responsible Use

- **Educational / research prototype only.**
- **Not for clinical diagnosis** or any treatment decision.
- The **ToxoPath-1 dataset is extremely small**; content built on it is illustrative.
- The **macro/micro model is unreliable** for general use — it was trained on about a dozen
  images, one of which was a mislabelled non-specimen photo (see Section 5A).
- The **PathMNIST models are domain-specific**: 64×64 colorectal H&E patches from one source.
  Other organs, stains, scanners or magnifications are out of domain.
- **Softmax / probability "confidence" is not clinical certainty.** A high score can still be
  wrong.
- **Out-of-domain images may be misclassified** with high confidence (for example, a ToxoPath
  image sent to the PathMNIST model).
- **Dataset routing is filename-based** (`pathmnist_` prefix), which is a convenience for the
  demo, not a content-based check.
- Any real clinical interpretation requires review by a qualified **pathologist**.

## 13. Rubric Mapping

| Rubric Requirement | Implementation | File | Status |
|---|---|---|---|
| UI layer — input form | Image upload with preview, analyze button, reset | `index.html` | ✅ |
| UI layer — result display | Result cards, confidence bars, dataset badge, Pathology Learning | `index.html` | ✅ |
| Service/API — POST endpoint | `POST /predict`, `POST /predict-pathology` | `app.py` | ✅ |
| Service/API — input validation | File present, extension, real PNG/JPEG, size limit → 400/413/415 | `services/validation.py` | ✅ |
| Service/API — result response | JSON with class, label, confidence, decision, explanation, dataset | `app.py` | ✅ |
| Business logic — output → decision | Confidence threshold → `classified` / `low_confidence`, labels, explanations | `services/business_logic.py` | ✅ |
| Business logic — separate module | Dedicated module, called by the API | `services/business_logic.py` | ✅ |
| Database — at least 2 tables | `analyses`, `predictions` | `db/schema.sql` | ✅ |
| Database — FOREIGN KEY | `predictions.analysis_id → analyses.id`, `ON DELETE CASCADE`, `PRAGMA foreign_keys = ON` | `db/schema.sql`, `services/database.py` | ✅ |
| Database — parameterized SQL, no f-string/concatenation | Constant SQL strings with `?` placeholders; AST test | `services/database.py`, `tests/test_api.py` | ✅ |
| Database — no hard-coded credentials | SQLite needs none; path from `DATABASE_PATH` env var | `config.py`, `.env.example` | ✅ |
| Model — trained/saved model file | `pathmnist_model.keras`, `pathmnist_logreg.pkl`, `macro2micro_model.pkl` | project root | ✅ |
| Model — API loads model and runs inference | Loaded once per process, reused | `services/model_loader.py` | ✅ |
| Evaluation on unseen test data | Official PathMNIST test split, untouched, leakage check | `train_compare_pathmnist.py`, `reports/` | ✅ |
| ≥ 2 models on the same task, compared | CNN vs. PCA + LogisticRegression, same 9-class task and test set | `train_compare_pathmnist.py`, `reports/model_comparison.md` | ✅ |
| Reproducible evaluation | Seed 42, pinned `requirements.txt`, identical results across runs | `train_compare_pathmnist.py`, `requirements.txt` | ✅ |
| No PII as model features | Features are image pixels only; no personal info collected; file name/image not stored | `services/`, `db/schema.sql` | ✅ |
| Automated tests | API, validation, DB/FK, SQL safety, business logic, frontend logic | `tests/` | ✅ |
