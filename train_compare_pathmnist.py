"""PathMNIST Model A (CNN) vs Model B (PCA + LogisticRegression) харьцуулалт.

Хоёр model ижил 9 ангиллын task дээр, training-д ашиглаагүй
ижил PathMNIST test split (бүтнээр нь) дээр үнэлэгдэнэ.

Дараалал:
  1. dataset load
  2. train/test separation verify
  3. reproducible stratified sampling (Model B)
  4. Model B training
  5. Model B save (pathmnist_logreg.pkl)
  6. CNN load (pathmnist_model.keras — дарж бичихгүй)
  7. same test set prediction
  8. metrics calculate
  9. comparison report save (reports/model_comparison.json, .md)
 10. confusion matrices save (reports/confusion_matrix_*.png)

Ажиллуулах:
    .venv-pathmnist/bin/python train_compare_pathmnist.py
"""

import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "pathmnist_64.npz"
CNN_PATH = BASE_DIR / "pathmnist_model.keras"
LOGREG_PATH = BASE_DIR / "pathmnist_logreg.pkl"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_B_TRAIN_SUBSET = 25_000
PCA_COMPONENTS = 100
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000
BATCH_SIZE = 512

CLASS_NAMES = [
    "Adipose",
    "Background",
    "Debris",
    "Lymphocytes",
    "Mucus",
    "Smooth muscle",
    "Normal colon mucosa",
    "Cancer-associated stroma",
    "Colorectal adenocarcinoma epithelium",
]

MODEL_A_NAME = "Model A — CNN"
MODEL_B_NAME = "Model B — PCA + LogisticRegression"


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hashes(images):
    return {hashlib.sha1(image.tobytes()).hexdigest() for image in images}


def to_features(images_uint8):
    """uint8 (N, 64, 64, 3) → float32 (N, 12288), CNN-тэй ижил /255 normalization."""
    return images_uint8.reshape(len(images_uint8), -1).astype(np.float32) / 255.0


def predict_in_batches(predict_fn, images_uint8):
    predictions = []
    for start in range(0, len(images_uint8), BATCH_SIZE):
        predictions.append(predict_fn(images_uint8[start:start + BATCH_SIZE]))
    return np.concatenate(predictions)


def compute_metrics(y_true, y_pred):
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": [
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in labels
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_confusion_matrix(matrix, title, path):
    matrix = np.array(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized (recall)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if normalized[i, j] > 0.5 else "#1f2937"
            ax.text(j, i, f"{matrix[i, j]}\n{normalized[i, j]:.0%}",
                    ha="center", va="center", fontsize=7.5, color=color)

    short_names = [f"{i} {name}" for i, name in enumerate(CLASS_NAMES)]
    ax.set_xticks(range(len(CLASS_NAMES)), short_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASS_NAMES)), short_names, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pick_winner(results):
    """Macro F1 (ангилал бүрийг ижил жинтэй) үндсэн шалгуур; тэнцвэл accuracy."""
    ranked = sorted(
        results.items(),
        key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]),
        reverse=True,
    )
    (winner, best), (runner_up, second) = ranked[0], ranked[1]
    return {
        "winner": winner,
        "runner_up": runner_up,
        "criterion": "macro_f1 (tie-break: accuracy)",
        "macro_f1_difference": best["macro_f1"] - second["macro_f1"],
        "accuracy_difference": best["accuracy"] - second["accuracy"],
    }


def pct(value):
    return f"{value * 100:.2f}%"


def write_markdown(report, path):
    a = report["models"][MODEL_A_NAME]
    b = report["models"][MODEL_B_NAME]
    data = report["data"]
    comparison = report["comparison"]
    leak = data["leakage_check"]

    lines = [
        "# PathMNIST Model Comparison",
        "",
        f"_Generated: {report['generated_at']} — `train_compare_pathmnist.py`, seed {report['seed']}_",
        "",
        "## Summary (untouched PathMNIST test set, n = "
        f"{data['test_size']:,})",
        "",
        "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
        "|---|---|---|---|---|",
    ]
    for name in (MODEL_A_NAME, MODEL_B_NAME):
        m = report["models"][name]
        lines.append(
            f"| {name} | {pct(m['accuracy'])} | {pct(m['macro_precision'])} | "
            f"{pct(m['macro_recall'])} | {pct(m['macro_f1'])} |"
        )

    lines += [
        "",
        f"**Winner: {comparison['winner']}** — selected by {comparison['criterion']}. "
        f"Macro F1 difference: {comparison['macro_f1_difference'] * 100:.2f} percentage points; "
        f"accuracy difference: {comparison['accuracy_difference'] * 100:.2f} percentage points.",
        "",
        "Model confidence / predicted probability is **not** used as a comparison metric.",
        "",
        "## Task",
        "",
        "Both models solve the same task: 9-class colorectal histopathology tissue "
        "classification (PathMNIST). The macro/micro SVM model in this project solves a "
        "different task and is **not** part of this comparison.",
        "",
        "## Dataset and splits",
        "",
        f"- Dataset file: `{data['dataset_file']}` (SHA-256 `{data['dataset_sha256'][:16]}…`)",
        f"- Image size: {data['image_shape']} (RGB, uint8)",
        f"- Official train split: {data['train_size']:,} images",
        f"- Official validation split: {data['val_size']:,} images (Model B: reported only, not used for fitting or tuning)",
        f"- Official test split: {data['test_size']:,} images — used **only** for final evaluation, "
        "identical for both models, not subsampled",
        "",
        "### Data leakage check",
        "",
        f"- Split arrays are separate keys in the `.npz` file; the test array is loaded only after both models are trained/loaded.",
        f"- Exact-duplicate images (SHA-1 of pixel bytes) between Model B training subset and test set: **{leak['subset_test_exact_duplicates']}**",
        f"- Exact-duplicate images between the full train split and test set: **{leak['train_test_exact_duplicates']}**",
        "- Hyperparameters of Model B are fixed in code (no search on the test set).",
        "",
        "## Model B training subset",
        "",
        f"- Model B is trained on a **stratified random subset of {data['model_b_train_subset']:,}** "
        f"of the {data['train_size']:,} training images (seed {report['seed']}), to keep RAM and training time manageable.",
        "- Model A (CNN) was trained earlier by `train_pathmnist.py` on the full training split.",
        "- Class distribution of the subset matches the full training split (stratified).",
        "",
        "| Class | Full train | Model B subset |",
        "|---|---|---|",
    ]
    for class_id, name in enumerate(CLASS_NAMES):
        lines.append(
            f"| {class_id} {name} | {data['train_class_counts'][class_id]:,} | "
            f"{data['subset_class_counts'][class_id]:,} |"
        )

    lines += [
        "",
        "## Preprocessing",
        "",
        "- Both models: pixel values scaled to [0, 1] by dividing by 255 (same as `app.py` inference).",
        "- Model A: input tensor 64×64×3.",
        "- Model B: image flattened to 12,288 features → StandardScaler → PCA.",
        "",
        "## Model architectures",
        "",
        f"**{MODEL_A_NAME}** (`pathmnist_model.keras`, loaded, not retrained)",
        "",
        "```",
        *a["architecture"],
        "```",
        "",
        f"**{MODEL_B_NAME}** (`pathmnist_logreg.pkl`)",
        "",
        "```",
        *b["architecture"],
        "```",
        "",
        "Memory/time optimizations for Model B: stratified training subset, float32 features, "
        "randomized SVD PCA (100 components), batched test-set inference.",
        f"Model B training time: {b['training_seconds']:.1f} s; "
        f"PCA explained variance: {b['pca_explained_variance']:.2%}; "
        f"validation accuracy (info only): {pct(b['validation_accuracy'])}.",
        "",
        "## Per-class metrics (test set)",
        "",
        "| Class | Support | A Precision | A Recall | A F1 | B Precision | B Recall | B F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row_a, row_b in zip(a["per_class"], b["per_class"]):
        lines.append(
            f"| {row_a['class_id']} {row_a['class_name']} | {row_a['support']:,} | "
            f"{row_a['precision']:.4f} | {row_a['recall']:.4f} | {row_a['f1']:.4f} | "
            f"{row_b['precision']:.4f} | {row_b['recall']:.4f} | {row_b['f1']:.4f} |"
        )

    lines += [
        "",
        "## Confusion matrices",
        "",
        "Rows = true label, columns = predicted label (counts and row-normalized %).",
        "",
        f"- Model A: [`confusion_matrix_cnn.png`](confusion_matrix_cnn.png)",
        f"- Model B: [`confusion_matrix_logreg.png`](confusion_matrix_logreg.png)",
        "",
        "## Limitations",
        "",
        "- Model B uses a training subset while Model A used the full training split, so the "
        "comparison also reflects training-data volume, not only model family.",
        "- Model A was trained earlier without fixed seeds; its evaluation here is deterministic, "
        "but retraining it would give slightly different weights.",
        "- PathMNIST train/validation patches come from NCT-CRC-HE-100K and the test patches from "
        "CRC-VAL-HE-7K (a separate patient cohort), all colorectal H&E at 64×64; performance on "
        "other organs, stains, scanners or magnifications is unknown.",
        "- Metrics describe patch-level tissue classification for education/research, not diagnosis.",
        "- Softmax / predicted probability is not a measure of clinical certainty.",
        "",
        "## Environment",
        "",
        f"- Python {report['environment']['python']}, numpy {report['environment']['numpy']}, "
        f"scikit-learn {report['environment']['scikit_learn']}, tensorflow {report['environment']['tensorflow']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.time()
    set_seeds()
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Dataset load (зөвхөн train/val; test-ийг 7-р алхамд ачаална)
    log(f"Loading {DATASET_PATH.name}")
    dataset_sha = sha256_file(DATASET_PATH)
    data = np.load(DATASET_PATH)
    expected_keys = {"train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"}
    if not expected_keys <= set(data.files):
        raise RuntimeError(f"Dataset keys missing: {expected_keys - set(data.files)}")

    train_images = data["train_images"]
    train_labels = data["train_labels"].ravel().astype(int)
    log(f"train_images {train_images.shape} {train_images.dtype}")

    # 3. Reproducible stratified subset
    all_indices = np.arange(len(train_labels))
    subset_indices, _ = train_test_split(
        all_indices,
        train_size=MODEL_B_TRAIN_SUBSET,
        stratify=train_labels,
        random_state=SEED,
    )
    subset_indices = np.sort(subset_indices)
    subset_images = train_images[subset_indices]
    subset_labels = train_labels[subset_indices]

    train_size = len(train_labels)
    train_class_counts = np.bincount(train_labels, minlength=9).tolist()
    subset_class_counts = np.bincount(subset_labels, minlength=9).tolist()
    train_hashes = image_hashes(train_images)
    subset_hashes = image_hashes(subset_images)
    del train_images  # RAM чөлөөлөх (~1.1 GB)

    # 4. Model B training (test data энд огт ороогүй)
    log(f"Training Model B on {len(subset_labels):,} images")
    model_b = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=SEED)),
        ("logreg", LogisticRegression(C=LOGREG_C, max_iter=LOGREG_MAX_ITER, random_state=SEED)),
    ])
    fit_started = time.time()
    model_b.fit(to_features(subset_images), subset_labels)
    training_seconds = time.time() - fit_started
    del subset_images
    log(f"Model B trained in {training_seconds:.1f}s")

    val_images = data["val_images"]
    val_labels = data["val_labels"].ravel().astype(int)
    val_accuracy = float(accuracy_score(
        val_labels, predict_in_batches(lambda x: model_b.predict(to_features(x)), val_images)
    ))
    val_size = len(val_labels)
    del val_images

    # 5. Model B save (шинэ файл; одоогийн model-уудыг дарж бичихгүй)
    joblib.dump(model_b, LOGREG_PATH)
    log(f"Saved {LOGREG_PATH.name}")

    # 6. CNN load
    import tensorflow as tf

    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    cnn = tf.keras.models.load_model(CNN_PATH)
    cnn_architecture = []
    cnn.summary(print_fn=lambda line, **_: cnn_architecture.append(line))

    # 2 + 7. Test set — энд анх удаа ачаалагдана; хоёр model-д яг ижил массив
    test_images = data["test_images"]
    test_labels = data["test_labels"].ravel().astype(int)
    test_hashes = image_hashes(test_images)
    leakage = {
        "train_test_exact_duplicates": len(train_hashes & test_hashes),
        "subset_test_exact_duplicates": len(subset_hashes & test_hashes),
    }
    if leakage["subset_test_exact_duplicates"] > 0:
        log(f"WARNING: {leakage['subset_test_exact_duplicates']} test images also in Model B training subset")
    log(f"Test set {test_images.shape}; leakage check {leakage}")

    test_fingerprint = hashlib.sha256(test_images.tobytes() + test_labels.tobytes()).hexdigest()

    log("Predicting test set with Model A (CNN)")
    pred_a = predict_in_batches(
        lambda x: np.argmax(cnn.predict(x.astype(np.float32) / 255.0, verbose=0), axis=1),
        test_images,
    )
    log("Predicting test set with Model B (PCA + LogReg)")
    pred_b = predict_in_batches(lambda x: model_b.predict(to_features(x)), test_images)

    if not (len(pred_a) == len(pred_b) == len(test_labels)):
        raise RuntimeError("Prediction length mismatch — test set not identical")

    # 8. Metrics
    metrics_a = compute_metrics(test_labels, pred_a)
    metrics_b = compute_metrics(test_labels, pred_b)
    for name, metrics in ((MODEL_A_NAME, metrics_a), (MODEL_B_NAME, metrics_b)):
        values = [metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")]
        values += [row[k] for row in metrics["per_class"] for k in ("precision", "recall", "f1")]
        if not all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
            raise RuntimeError(f"{name}: metric out of range")

    pca = model_b.named_steps["pca"]
    metrics_a.update({
        "file": CNN_PATH.name,
        "trained_on": f"full train split ({train_size:,} images) by train_pathmnist.py",
        "architecture": cnn_architecture,
        "test_predictions_fingerprint": hashlib.sha256(pred_a.astype(np.int64).tobytes()).hexdigest(),
    })
    metrics_b.update({
        "file": LOGREG_PATH.name,
        "trained_on": f"stratified subset of {MODEL_B_TRAIN_SUBSET:,} train images (seed {SEED})",
        "architecture": [
            "Pipeline:",
            "  1. StandardScaler()",
            f"  2. PCA(n_components={PCA_COMPONENTS}, svd_solver='randomized', random_state={SEED})",
            f"  3. LogisticRegression(C={LOGREG_C}, max_iter={LOGREG_MAX_ITER}, random_state={SEED})"
            "  # multinomial (softmax) for 9 classes",
            f"Input: 64x64x3 image flattened to {64 * 64 * 3:,} features",
        ],
        "training_seconds": training_seconds,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "logreg_iterations": int(np.max(model_b.named_steps["logreg"].n_iter_)),
        "validation_accuracy": val_accuracy,
        "test_predictions_fingerprint": hashlib.sha256(pred_b.astype(np.int64).tobytes()).hexdigest(),
    })

    results = {MODEL_A_NAME: metrics_a, MODEL_B_NAME: metrics_b}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "task": "PathMNIST 9-class colorectal histopathology tissue classification",
        "class_names": CLASS_NAMES,
        "data": {
            "dataset_file": DATASET_PATH.name,
            "dataset_sha256": dataset_sha,
            "image_shape": list(test_images.shape[1:]),
            "train_size": train_size,
            "val_size": val_size,
            "test_size": int(len(test_labels)),
            "test_set_fingerprint": test_fingerprint,
            "test_set_subsampled": False,
            "model_b_train_subset": MODEL_B_TRAIN_SUBSET,
            "train_class_counts": train_class_counts,
            "subset_class_counts": subset_class_counts,
            "test_class_counts": np.bincount(test_labels, minlength=9).tolist(),
            "leakage_check": leakage,
        },
        "models": results,
        "comparison": pick_winner({name: m for name, m in results.items()}),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "tensorflow": tf.__version__,
        },
        "runtime_seconds": None,
    }

    # 10. Confusion matrices
    save_confusion_matrix(metrics_a["confusion_matrix"], f"{MODEL_A_NAME} — PathMNIST test set",
                          REPORTS_DIR / "confusion_matrix_cnn.png")
    save_confusion_matrix(metrics_b["confusion_matrix"], f"{MODEL_B_NAME} — PathMNIST test set",
                          REPORTS_DIR / "confusion_matrix_logreg.png")

    # 9. Reports (бүх зүйл амжилттай болсны дараа л бичнэ)
    report["runtime_seconds"] = round(time.time() - started, 1)
    (REPORTS_DIR / "model_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, REPORTS_DIR / "model_comparison.md")

    log("Done")
    for name, m in results.items():
        log(f"{name}: accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    log(f"Winner: {report['comparison']['winner']}")


if __name__ == "__main__":
    main()
