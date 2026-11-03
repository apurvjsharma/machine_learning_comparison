"""
Streamlit front-end for the Online Shoppers Purchasing Intention classifiers.

The app is a pure inference client: it loads the five pipelines that
`train_models.py` already fitted and saved under model/, scores whatever test
CSV is uploaded, and reports the metrics. Nothing is re-fitted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
BUNDLED_TEST_CSV = PROJECT_ROOT / "test_data.csv"

CLASS_LABELS = ["No purchase", "Purchase"]
PALETTE = {
    "ink": "#1b2a4a",
    "accent": "#2f6f6b",
    "warm": "#c8763c",
}

st.set_page_config(
    page_title="Shopper Intent Classifier Bench",
    page_icon="🛒",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1200px; }
      .bench-title {
          font-size: 2.1rem; font-weight: 700; color: #1b2a4a;
          letter-spacing: -0.02em; margin-bottom: 0.1rem;
      }
      .bench-sub { color: #5c6b84; font-size: 0.95rem; margin-bottom: 1.4rem; }
      .metric-card {
          border: 1px solid #e2e6ee; border-left: 4px solid #2f6f6b;
          border-radius: 6px; padding: 0.7rem 0.9rem; background: #fbfcfe;
      }
      .metric-card .label {
          font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em;
          color: #6b7a93; white-space: nowrap;
      }
      .metric-card .value {
          font-size: 1.3rem; font-weight: 650; color: #1b2a4a; line-height: 1.3;
          white-space: nowrap;
      }
      .note { color: #6b7a93; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Loading of the saved artefacts
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_schema() -> dict:
    return json.loads((MODEL_DIR / "schema.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_pipelines() -> dict:
    """Deserialise one fitted sklearn Pipeline per classifier."""
    schema = load_schema()
    bench = {}
    for display_name, filename in schema["model_files"].items():
        path = MODEL_DIR / filename
        if path.exists():
            bench[display_name] = joblib.load(path)
    return bench


@st.cache_data(show_spinner=False)
def load_reference_metrics() -> dict:
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------
def confidence_scores(pipeline, features: pd.DataFrame) -> np.ndarray:
    """Positive-class score. SVC has no predict_proba, so use its margin."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(features)[:, 1]
    return pipeline.decision_function(features)


def six_metrics(truth, predictions, scores) -> dict:
    return {
        "Accuracy": accuracy_score(truth, predictions),
        "AUC": roc_auc_score(truth, scores),
        "Precision": precision_score(truth, predictions, zero_division=0),
        "Recall": recall_score(truth, predictions, zero_division=0),
        "F1": f1_score(truth, predictions, zero_division=0),
        "MCC": matthews_corrcoef(truth, predictions),
    }


def normalise_upload(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Coerce the two boolean-ish columns so True/False and 1/0 both work."""
    frame = frame.copy()
    truthy = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
    for column in ("Weekend", schema["target"]):
        if column in frame.columns and frame[column].dtype == object:
            frame[column] = (
                frame[column].astype(str).str.strip().str.lower().map(truthy)
            )
    for column in ("Weekend", schema["target"]):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame


def validate_upload(frame: pd.DataFrame, schema: dict) -> list[str]:
    """Return a list of human-readable problems; empty list means the CSV is fine."""
    problems = []
    missing = [c for c in schema["feature_order"] if c not in frame.columns]
    if missing:
        problems.append("Missing feature column(s): " + ", ".join(missing))
    if schema["target"] not in frame.columns:
        problems.append(f"Missing target column '{schema['target']}'.")
    elif frame[schema["target"]].isna().any():
        problems.append(f"Target column '{schema['target']}' has unreadable values.")
    elif frame[schema["target"]].nunique() < 2:
        problems.append(
            f"Target column '{schema['target']}' holds a single class, so AUC and "
            "MCC cannot be computed. Upload a CSV containing both classes."
        )
    if frame.empty:
        problems.append("The uploaded file has no rows.")
    return problems


def metric_card(column, label: str, value: float) -> None:
    column.markdown(
        f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value:.4f}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_confusion(matrix: np.ndarray, title: str):
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="BuGn",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title, fontsize=10, color=PALETTE["ink"])
    figure.tight_layout()
    return figure


def plot_roc(truth, scores, label: str):
    false_pos, true_pos, _ = roc_curve(truth, scores)
    area = roc_auc_score(truth, scores)
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    axis.plot(false_pos, true_pos, color=PALETTE["accent"], lw=2, label=f"AUC = {area:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="#b6bfcf", lw=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"ROC - {label}", fontsize=10, color=PALETTE["ink"])
    axis.legend(loc="lower right", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


def plot_metric_bars(table: pd.DataFrame, metric: str):
    ordered = table.sort_values(metric, ascending=True)
    figure, axis = plt.subplots(figsize=(6.6, 3.2))
    axis.barh(ordered.index, ordered[metric], color=PALETTE["accent"], height=0.6)
    for y, value in enumerate(ordered[metric]):
        axis.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=8.5,
                  color=PALETTE["ink"])
    axis.set_xlim(0, 1.12)
    axis.set_xlabel(metric)
    axis.set_title(f"{metric} across the five classifiers", fontsize=10,
                   color=PALETTE["ink"])
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
schema = load_schema()
bench = load_pipelines()
reference = load_reference_metrics()

st.markdown('<div class="bench-title">Shopper Intent Classifier Bench</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="bench-sub">Five pre-trained classifiers scored on an uploaded '
    'test split of the UCI Online Shoppers Purchasing Intention dataset. '
    'The models are loaded from disk and only used for inference - no retraining '
    'happens in this app.</div>',
    unsafe_allow_html=True,
)

if not bench:
    st.error(
        "No model files were found under `model/`. Run `python train_models.py` "
        "to generate them before starting the app."
    )
    st.stop()

with st.sidebar:
    st.header("Test data")
    upload = st.file_uploader("Upload a test CSV", type="csv")
    use_bundled = st.checkbox(
        "Use the bundled test_data.csv",
        value=True,
        help="The 2,441-row stratified hold-out split produced by train_models.py.",
    )

    st.header("Model")
    selected_model = st.selectbox("Classifier to inspect", list(bench.keys()))

    st.header("Decision threshold")
    threshold = st.slider(
        "Positive-class cut-off",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help=(
            "Applies to the four probability-based models. The SVM predicts from "
            "the sign of its margin and ignores this slider."
        ),
    )

    st.caption(
        f"Loaded {len(bench)} pipelines from `model/`. "
        f"Target column: `{schema['target']}`."
    )

# Resolve which CSV to score.
if upload is not None:
    test_frame = pd.read_csv(upload)
    source_label = f"uploaded file - {upload.name}"
elif use_bundled and BUNDLED_TEST_CSV.exists():
    test_frame = pd.read_csv(BUNDLED_TEST_CSV)
    source_label = "bundled test_data.csv"
else:
    st.info("Upload a test CSV in the sidebar, or tick *Use the bundled test_data.csv*.")
    st.stop()

test_frame = normalise_upload(test_frame, schema)
problems = validate_upload(test_frame, schema)
if problems:
    st.error("The CSV cannot be scored:\n\n" + "\n".join(f"- {p}" for p in problems))
    st.stop()

x_test = test_frame[schema["feature_order"]]
y_test = test_frame[schema["target"]].astype(int)

head = st.columns(4)
head[0].metric("Rows scored", f"{len(test_frame):,}")
head[1].metric("Input features", len(schema["feature_order"]))
head[2].metric("Purchase rate", f"{y_test.mean():.2%}")
head[3].metric("Source", source_label.split(" - ")[0].title())

tab_single, tab_compare, tab_data = st.tabs(
    ["Selected model", "All five models", "Data preview"]
)

# --- Tab 1: one model in detail ------------------------------------------
with tab_single:
    pipeline = bench[selected_model]
    scores = confidence_scores(pipeline, x_test)

    if hasattr(pipeline, "predict_proba"):
        predictions = (scores >= threshold).astype(int)
        threshold_note = f"Threshold applied: {threshold:.2f}"
    else:
        predictions = pipeline.predict(x_test).astype(int)
        threshold_note = "Predicted from the sign of the SVM margin (slider not used)."

    st.subheader(f"{selected_model} - evaluation metrics")
    st.markdown(f'<span class="note">{threshold_note}</span>', unsafe_allow_html=True)

    live = six_metrics(y_test, predictions, scores)
    cards = st.columns(6)
    for card, (label, value) in zip(cards, live.items()):
        metric_card(card, label, value)

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion(confusion_matrix(y_test, predictions), selected_model))
    with right:
        st.markdown("**ROC curve**")
        st.pyplot(plot_roc(y_test, scores, selected_model))

    st.markdown("**Classification report**")
    report = classification_report(
        y_test, predictions, target_names=CLASS_LABELS, output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")

    predicted_frame = test_frame.copy()
    predicted_frame["predicted_Revenue"] = predictions
    predicted_frame["positive_score"] = np.round(scores, 6)
    st.download_button(
        "Download predictions as CSV",
        predicted_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# --- Tab 2: the comparison table -----------------------------------------
with tab_compare:
    st.subheader("Comparison table on the current test data")
    st.markdown(
        '<span class="note">Every row is computed live from the CSV above using '
        'each saved pipeline at its own default decision rule, so this table is '
        'reproducible by re-uploading the same file.</span>',
        unsafe_allow_html=True,
    )

    rows = {}
    for name, candidate in bench.items():
        candidate_scores = confidence_scores(candidate, x_test)
        rows[name] = six_metrics(y_test, candidate.predict(x_test).astype(int),
                                 candidate_scores)
    comparison = pd.DataFrame(rows).transpose()

    st.dataframe(
        comparison.style.format("{:.4f}").background_gradient(cmap="BuGn", axis=0),
        width="stretch",
    )

    winner_metric = st.selectbox(
        "Rank the models by", list(comparison.columns), index=5,
    )
    st.pyplot(plot_metric_bars(comparison, winner_metric))

    best = comparison[winner_metric].idxmax()
    st.success(
        f"Best {winner_metric} on this test data: **{best}** "
        f"({comparison.loc[best, winner_metric]:.4f})"
    )

    if reference:
        with st.expander("Reference metrics recorded at training time"):
            st.caption(
                f"Held-out split of {reference.get('n_test')} sessions, "
                f"random_state={reference.get('random_state')}. These are the "
                "numbers quoted in the README."
            )
            st.dataframe(
                pd.DataFrame(reference["metrics"]).transpose().round(4),
                width="stretch",
            )

# --- Tab 3: what was actually loaded -------------------------------------
with tab_data:
    st.subheader("First rows of the scored CSV")
    st.dataframe(test_frame.head(25), width="stretch")

    st.subheader("Class balance")
    balance = (
        y_test.map({0: CLASS_LABELS[0], 1: CLASS_LABELS[1]})
        .value_counts()
        .rename("sessions")
        .to_frame()
    )
    balance["share"] = (balance["sessions"] / len(y_test)).map("{:.2%}".format)
    st.dataframe(balance, width="stretch")

    st.subheader("Expected columns")
    st.json(
        {
            "numeric_features": schema["numeric_features"],
            "categorical_features": schema["categorical_features"],
            "target": schema["target"],
        }
    )
