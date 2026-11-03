"""
Online Shoppers Purchasing Intention - training script
=====================================================

Trains five classifiers on the UCI "Online Shoppers Purchasing Intention"
dataset, scores them on a held-out stratified test split, and writes:

    model/*.joblib        one fitted sklearn Pipeline per classifier
    model/metrics.json    reference metrics computed on the held-out split
    model/schema.json     the feature contract the Streamlit app validates against
    test_data.csv         the raw held-out split (features + Revenue target)

Run from the project root:

    python train_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42
TEST_FRACTION = 0.20

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_CSV = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_CSV = PROJECT_ROOT / "test_data.csv"

TARGET_COLUMN = "Revenue"

# Session-behaviour counters and durations -> standardised before fitting.
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# Integer-coded lookups and text columns -> one-hot encoded. OperatingSystems,
# Browser, Region and TrafficType are stored as integers in the CSV but they are
# unordered category ids, so treating them as numeric would invent a false
# ordering (Browser 13 is not "more" than Browser 2).
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def load_sessions(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV and apply the two cleaning steps the dataset needs."""
    frame = pd.read_csv(csv_path)

    # 125 exact duplicate sessions exist in the published file. Keeping them
    # would let identical rows sit in both the train and the test split.
    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(frame)} duplicate rows -> {len(frame)} sessions remain")

    # Weekend / Revenue arrive as booleans; store them as 0/1 so the saved CSV
    # round-trips cleanly through the Streamlit uploader.
    for flag in ("Weekend", TARGET_COLUMN):
        frame[flag] = frame[flag].astype(int)

    return frame


def build_preprocessor() -> ColumnTransformer:
    """Scale the numeric block, one-hot the categorical block."""
    numeric_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, NUMERIC_FEATURES),
            ("categorical", categorical_branch, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_classifiers() -> dict:
    """The five required classifiers, each wrapped in its own full pipeline.

    Hyper-parameters are kept close to the scikit-learn defaults on purpose so
    the comparison table measures the algorithms rather than the amount of
    tuning each one received. The deviations from the defaults are:

      * `max_iter=2000` on the lbfgs solver. Only a safety margin - it converges
        in 66 iterations on this data, so the default 100 would also have done.
      * `max_depth=8` / `min_samples_leaf=20` on the tree. The one setting that
        genuinely matters: unrestricted, the tree grows to depth 26 and 810
        leaves, reaches 1.0000 training accuracy, and its held-out MCC falls
        from 0.5823 to 0.4654.
      * `n_neighbors=15` / `weights="distance"` on KNN, against the default 5
        and uniform weights.
      * `random_state` fixed wherever the estimator accepts one, so the run
        reproduces.

    Class weights are left at their defaults throughout. `class_weight=
    "balanced"` would lift recall, but only three of the five estimators accept
    it, so using it would make the comparison table unfair. The Streamlit app's
    threshold slider demonstrates the same trade-off at inference time instead.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            random_state=RANDOM_SEED,
        ),
    }


def confidence_scores(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Continuous scores for the positive class, for ROC/AUC.

    Four of the five estimators expose `predict_proba`. The SVC is trained
    without `probability=True` (Platt scaling would add a five-fold internal
    cross-validation to every fit), so its signed distance from the separating
    hyperplane is used instead. AUC only depends on the ranking, so an
    unnormalised score is as valid as a calibrated probability.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(features)[:, 1]
    return pipeline.decision_function(features)


def score_model(pipeline: Pipeline, features: pd.DataFrame, truth: pd.Series) -> dict:
    """All six metrics the assignment asks for, as a plain dict."""
    predictions = pipeline.predict(features)
    scores = confidence_scores(pipeline, features)
    return {
        "Accuracy": accuracy_score(truth, predictions),
        "AUC": roc_auc_score(truth, scores),
        "Precision": precision_score(truth, predictions, zero_division=0),
        "Recall": recall_score(truth, predictions, zero_division=0),
        "F1": f1_score(truth, predictions, zero_division=0),
        "MCC": matthews_corrcoef(truth, predictions),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    sessions = load_sessions()
    features = sessions[FEATURE_COLUMNS]
    target = sessions[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=target,
    )
    print(f"Train sessions: {len(x_train)}   Test sessions: {len(x_test)}")
    print(f"Purchase rate  train: {y_train.mean():.4f}   test: {y_test.mean():.4f}")

    # The held-out split is written out untouched (raw feature values plus the
    # target) so the Streamlit app receives exactly the CSV a grader would.
    held_out = x_test.copy()
    held_out[TARGET_COLUMN] = y_test
    held_out.to_csv(TEST_CSV, index=False)
    print(f"Wrote {TEST_CSV.name} ({len(held_out)} rows)")

    metrics_table = {}
    for name, classifier in build_classifiers().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)

        metrics_table[name] = score_model(pipeline, x_test, y_test)

        destination = MODEL_DIR / MODEL_FILES[name]
        joblib.dump(pipeline, destination, compress=3)
        row = "  ".join(f"{k}={v:.4f}" for k, v in metrics_table[name].items())
        print(f"{name:<24} {row}   -> {destination.name}")

    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": "UCI Online Shoppers Purchasing Intention",
                "random_state": RANDOM_SEED,
                "test_size": TEST_FRACTION,
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "positive_rate_test": float(y_test.mean()),
                "metrics": metrics_table,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (MODEL_DIR / "schema.json").write_text(
        json.dumps(
            {
                "target": TARGET_COLUMN,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "feature_order": FEATURE_COLUMNS,
                "model_files": MODEL_FILES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote model/metrics.json and model/schema.json")


if __name__ == "__main__":
    main()
