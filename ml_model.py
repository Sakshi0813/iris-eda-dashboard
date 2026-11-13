"""Train, evaluate, save, and load beginner-friendly Iris classifiers.

Run ``python ml_model.py`` from the project folder to train both models.
The source CSV is read only; all 150 rows, including duplicate rows, are kept.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

# Keep small dataset operations reliable on machines that expose many CPU cores
# but have limited memory available for OpenBLAS worker threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "iris.csv"
MODEL_DIR = PROJECT_DIR / "models"
FEATURES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET = "species"
RANDOM_STATE = 42
ARTIFACT_VERSION = 1
MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
}


def file_sha256(path: Path) -> str:
    """Return a content hash used to detect dataset changes after training."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_iris_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the project CSV without changing or deduplicating it."""
    data = pd.read_csv(csv_path)
    missing_columns = [column for column in [*FEATURES, TARGET] if column not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if data.empty:
        raise ValueError("The Iris dataset is empty.")

    data = data[[*FEATURES, TARGET]].copy()
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data[TARGET] = data[TARGET].astype("string").str.strip().str.replace("^Iris-", "", regex=True).str.title()
    data.loc[data[TARGET] == "", TARGET] = pd.NA
    if data[TARGET].isna().any():
        raise ValueError("Every training row must have a species label.")
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Numerical features must contain finite values; fix missing or invalid CSV values first.")
    if data[TARGET].nunique() < 2:
        raise ValueError("At least two species are needed to train a classifier.")
    return data


def make_train_test_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible stratified 80/20 split, keeping exact duplicates together.

    Groups are built from all four features plus the species label. This avoids
    placing an exact duplicate record in both train and test while keeping every
    source row in one of the two sets. StratifiedGroupKFold makes five folds;
    the first fold is the 20% test set.
    """
    records = data[[*FEATURES, TARGET]].astype(str)
    groups = pd.factorize(pd.MultiIndex.from_frame(records))[0]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_indices, test_indices = next(
        splitter.split(data[FEATURES], data[TARGET], groups=groups)
    )
    if set(groups[train_indices]).intersection(set(groups[test_indices])):
        raise RuntimeError("An exact duplicate group crossed the train/test split.")
    return train_indices, test_indices


def build_pipeline(model_name: str) -> Pipeline:
    """Create the requested classifier with preprocessing inside the pipeline."""
    if model_name == "Decision Tree":
        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=1
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Calculate multiclass metrics and readable per-class evaluation details."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=np.arange(len(labels)), target_names=labels,
            output_dict=True, zero_division=0,
        ),
        "classification_report_text": classification_report(
            y_true, y_pred, labels=np.arange(len(labels)), target_names=labels,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(len(labels))
        ).tolist(),
        "class_labels": labels,
    }


def train_models(csv_path: Path = DATA_PATH, model_dir: Path = MODEL_DIR) -> dict[str, dict[str, Any]]:
    """Train both models on the training fold and save reusable local artifacts."""
    data = load_iris_data(csv_path)
    train_indices, test_indices = make_train_test_split(data)
    x_train = data.iloc[train_indices][FEATURES]
    x_test = data.iloc[test_indices][FEATURES]
    y_train_text = data.iloc[train_indices][TARGET].astype(str)
    y_test_text = data.iloc[test_indices][TARGET].astype(str)

    # Fit label encoding on training labels only, then reuse it for the test labels.
    label_encoder = LabelEncoder().fit(y_train_text)
    y_train = label_encoder.transform(y_train_text)
    try:
        y_test = label_encoder.transform(y_test_text)
    except ValueError as error:
        raise ValueError("The training split must contain every species in the test split.") from error

    model_dir.mkdir(parents=True, exist_ok=True)
    data_hash = file_sha256(csv_path)
    source_bounds = {
        feature: (float(data[feature].min()), float(data[feature].max()))
        for feature in FEATURES
    }
    results: dict[str, dict[str, Any]] = {}
    for model_name, filename in MODEL_FILES.items():
        pipeline = build_pipeline(model_name)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = evaluate_predictions(y_test, predictions, list(label_encoder.classes_))
        artifact = {
            "artifact_version": ARTIFACT_VERSION,
            "model_name": model_name,
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "features": FEATURES.copy(),
            "metrics": metrics,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "dataset_rows": int(len(data)),
            "duplicate_rows_retained": int(data.duplicated().sum()),
            "source_sha256": data_hash,
            "input_bounds": source_bounds,
        }
        joblib.dump(artifact, model_dir / filename)
        results[model_name] = artifact
    return results


def load_saved_model(model_name: str, model_dir: Path = MODEL_DIR, csv_path: Path = DATA_PATH) -> dict[str, Any]:
    """Load one known project model and reject missing, stale, or malformed files.

    Joblib files use Python serialization. Only load model files created by this
    project; do not load artifacts supplied by an unknown source.
    """
    if model_name not in MODEL_FILES:
        raise ValueError(f"Choose one of the supported models: {', '.join(MODEL_FILES)}")
    path = (model_dir / MODEL_FILES[model_name]).resolve()
    if path.parent != model_dir.resolve():
        raise ValueError("The model path is outside the project models folder.")
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}. Run `python ml_model.py` first.")
    artifact = joblib.load(path)
    required = {"artifact_version", "model_name", "pipeline", "label_encoder", "features", "metrics", "source_sha256", "input_bounds"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("The saved model file is not a valid project model artifact.")
    if artifact["artifact_version"] != ARTIFACT_VERSION or artifact["model_name"] != model_name:
        raise ValueError("The saved model version or model name is invalid.")
    if artifact["features"] != FEATURES:
        raise ValueError("The saved model feature order does not match this project.")
    if not csv_path.is_file() or file_sha256(csv_path) != artifact["source_sha256"]:
        raise ValueError("The Iris CSV has changed since training. Run `python ml_model.py` to retrain.")
    return artifact


def predict_species(artifact: dict[str, Any], values: dict[str, float]) -> str:
    """Predict a species for one measurement row after validating its inputs."""
    if set(values) != set(FEATURES):
        raise ValueError("Provide exactly the four Iris measurement features.")
    row = {}
    for feature in FEATURES:
        try:
            value = float(values[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature} must be a number.") from error
        if not np.isfinite(value):
            raise ValueError(f"{feature} must be a finite number.")
        lower, upper = artifact["input_bounds"][feature]
        if value < lower or value > upper:
            raise ValueError(f"{feature} must be between {lower:g} and {upper:g} cm for this dataset.")
        row[feature] = value
    encoded_prediction = artifact["pipeline"].predict(pd.DataFrame([row], columns=FEATURES))
    return str(artifact["label_encoder"].inverse_transform(encoded_prediction)[0])


def main() -> None:
    """Train both classifiers, save artifacts, and print their test metrics."""
    models = train_models()
    print(f"Trained on {next(iter(models.values()))['train_rows']} rows; evaluated on {next(iter(models.values()))['test_rows']} rows.")
    print("Exact duplicate rows were retained and kept together in the split.")
    for model_name, artifact in models.items():
        metrics = artifact["metrics"]
        print(f"\n{model_name}")
        print(
            f"Accuracy: {metrics['accuracy']:.3f} | "
            f"Macro precision: {metrics['precision_macro']:.3f} | "
            f"Macro recall: {metrics['recall_macro']:.3f} | "
            f"Macro F1: {metrics['f1_macro']:.3f}"
        )
        print(metrics["classification_report_text"])
        print("Confusion matrix (actual rows × predicted columns):")
        print(np.asarray(metrics["confusion_matrix"]))
    print(f"\nSaved model files in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
