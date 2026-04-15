# --------------------------------------------------------------------------------------------------------
# EXPERIMENT NOTES (TRANSDUCTIVE THRESHOLD TUNING — APPROACH C)
# --------------------------------------------------------------------------------------------------------
#
# Goal:
# - Use Isolation Forest in a TRANSDUCTIVE setting: fit on the ENTIRE dataset,
#   then use a small labeled subset to calibrate the anomaly score threshold.
#
# Key decisions made in this file:
# 1) Fit Isolation Forest on ALL data (no train/test split).
#    The model learns the full density structure of the dataset.
# 2) Generate continuous anomaly scores via score_samples() for every sample.
# 3) Use ground-truth labels ONLY to find the optimal decision threshold:
#    sweep the precision-recall curve and pick the threshold that maximizes F1.
# 4) Apply that threshold to produce final binary predictions.
# 5) Evaluate with imbalance-aware metrics: PR-AUC, F1, precision, recall,
#    precision@k, recall@k.
#
# Why transductive (Approach C) instead of inductive (main_experiment.py):
# - IF sees 100% of data -> better density estimation than 75% train split.
# - Labels are used only for ONE scalar (threshold), not for model training
#   or hyperparameter grid search -> minimal risk of overfitting.
# - Simpler pipeline: no train/val/test split machinery needed.
# - Consistent with team agreement on transductive modeling.
#
# --------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------
# Utilities

def verb_aware_print(msg, verb=True):
    if verb and msg: print(msg)

def noop_callback(**kwargs):
    pass

# --------------------------------------------------------------------------------------------------------

# Imports and constants

import kagglehub
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
VERBOSE = True

# Default Isolation Forest settings for the transductive fit.
# contamination="auto" means IF uses its own offset (0.5) for predict(),
# but we override predictions with our own threshold anyway.
DEFAULT_IF_PARAMS = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": "auto",
}


# 1. Load data

def load_data():
    """
    Load the credit card fraud dataset and separate features from the target variable.
    """

    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_file = Path(dataset_path) / "creditcard.csv"
    df = pd.read_csv(csv_file)

    assert "Class" in df.columns, "Target variable 'Class' not found in the dataset."

    # Remove duplicates BEFORE separating the target to keep features/labels synced
    initial_shape = df.shape
    df = df.drop_duplicates(keep="first")
    if initial_shape[0] - df.shape[0] > 0:
        print(f"\n[load_data] Safely dropped {initial_shape[0] - df.shape[0]} duplicate rows from the source dataframe.")

    data = df.drop(columns=["Class"])
    target = df["Class"]

    assert "Class" not in data.columns, "Target variable 'Class' should not be part of the data."

    return data, target


# 2.1 Cleaning

def report_data_quality(df: pd.DataFrame, title: str) -> None:
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    print(f"\n===== {title} =====")
    print("Shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Total missing values:", missing.sum())
    if len(missing_cols) == 0:
        print("Columns with missing values: 0")
    else:
        print(f"Columns with missing values: {len(missing_cols)}")
        for col, count in missing_cols.items():
            print(f"  - {col}: {count}")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.isna().sum().sum() == 0:
        print("\nNo missing values found. No imputation needed.")
        return df

    cleaned = df.copy()

    # Numeric features -> median
    num_cols = cleaned.select_dtypes("number").columns
    cleaned[num_cols] = cleaned[num_cols].fillna(cleaned[num_cols].median())

    # Object features -> mode, fallback to "unknown"
    obj_cols = cleaned.select_dtypes(exclude="number").columns
    for col in obj_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode(dropna=True).iloc[0] if not cleaned[col].mode(dropna=True).empty else "unknown")

    print("Filled missing values: numeric -> median, categorical -> mode.")
    return cleaned

def clean_data(data):
    """
    Perform necessary data cleaning steps using Sufyan's logic.
    Note: We do not drop duplicates here because the labels (target) are already separated from 'data'.
    Dropping rows here would desync 'data' from 'target'. 
    If duplicates need removing, it must be done before separating the Class column.
    """
    df = data.copy()
    report_data_quality(df, "Before Cleaning")
    df = handle_missing_values(df)
    report_data_quality(df, "After Cleaning")
    return df


# 2.2 Preprocessing

def preprocess_data(data):
    """
    Preprocess the cleaned data to make it suitable for modeling with Isolation Forest.
    Scale Time and Amount (V1..V28 are already PCA-transformed).
    """

    df = data.copy()

    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    X = df.to_numpy()
    return X


# 3. Modeling (Transductive)

def modeling(X, if_params=None):
    """
    Fit Isolation Forest on the FULL dataset (transductive approach).

    Unlike the inductive approach in main_experiment.py, we do NOT split the data.
    The model sees every sample so it can learn the complete density structure.

    Parameters
    ----------
    X : np.ndarray
        The full preprocessed dataset.
    if_params : dict, optional
        Override default IF hyperparameters.

    Returns
    -------
    model : IsolationForest
        Fitted model.
    """
    params = {**DEFAULT_IF_PARAMS, "random_state": RANDOM_STATE, "n_jobs": -1}
    if if_params:
        params.update(if_params)
    model = IsolationForest(**params).fit(X)
    return model


# 4.1 Scoring

def score_samples(model, X):
    """
    Generate continuous anomaly scores for all samples.

    Uses -score_samples() so that HIGHER values = MORE anomalous.
    This is the raw output before any thresholding.

    Parameters
    ----------
    model : IsolationForest
        Fitted model.
    X : np.ndarray
        Data to score.

    Returns
    -------
    anomaly_scores : np.ndarray
        Anomaly scores (higher = more anomalous).
    """
    anomaly_scores = -model.score_samples(X)  # higher means "more anomalous"
    return anomaly_scores


# 4.2 Threshold Tuning (Post-hoc, using labeled data ONLY)

def find_optimal_threshold(anomaly_scores, labels, metric="f1"):
    """
    Find the anomaly score threshold that maximizes F1 on the labeled subset.

    This is the ONLY place where ground-truth labels influence the pipeline.
    The model was already fitted without seeing labels — we only use them here
    to pick a single scalar decision boundary.

    Approach:
    - Compute precision-recall curve across all possible thresholds.
    - For each threshold, compute F1 = 2 * precision * recall / (precision + recall).
    - Return the threshold that yields the highest F1.

    Parameters
    ----------
    anomaly_scores : np.ndarray
        Anomaly scores from the fitted model (higher = more anomalous).
    labels : np.ndarray
        Ground-truth labels (0 = normal, 1 = anomaly/fraud).
    metric : str
        Currently only "f1" is supported.

    Returns
    -------
    best_threshold : float
        The optimal score cutoff.
    threshold_info : dict
        Dictionary with best F1, precision, recall, and the threshold value.
    """
    labels = np.asarray(labels).astype(int)
    anomaly_scores = np.asarray(anomaly_scores)

    precisions, recalls, thresholds = precision_recall_curve(labels, anomaly_scores)

    # Compute F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    # The last precision/recall pair corresponds to threshold=max(scores)+1 (all negative)
    best_idx = np.argmax(f1_scores[:-1])  # exclude the last artificial point
    best_threshold = float(thresholds[best_idx])

    threshold_info = {
        "best_f1": float(f1_scores[best_idx]),
        "best_precision": float(precisions[best_idx]),
        "best_recall": float(recalls[best_idx]),
        "threshold": best_threshold,
        "n_thresholds_evaluated": len(thresholds),
    }

    return best_threshold, threshold_info


# 4.3 Prediction (using the optimized threshold)

def predict_with_threshold(anomaly_scores, threshold):
    """
    Apply a custom threshold to anomaly scores to produce binary predictions.

    Parameters
    ----------
    anomaly_scores : np.ndarray
        Anomaly scores (higher = more anomalous).
    threshold : float
        Decision boundary. Scores >= threshold -> anomaly (1), below -> normal (0).

    Returns
    -------
    y_pred : np.ndarray
        Binary predictions (0 = normal, 1 = anomaly).
    """
    y_pred = (anomaly_scores >= threshold).astype(int)
    return y_pred


# 5. Evaluation

def evaluate(target, y_pred, anomaly_scores, top_k_ratio=0.01):
    """
    Evaluate with metrics appropriate for class imbalance.

    Reported metrics:
    - pr_auc: primary metric for rare-event ranking quality
    - precision / recall / f1: threshold-dependent quality at our chosen cutoff
    - predicted_anomaly_rate: operational signal (alert volume)
    - precision_at_k / recall_at_k: quality under a fixed review budget
    - pr_curve_precision / pr_curve_recall: arrays for plotting PR curve
    """

    y_true = np.asarray(target).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    anomaly_scores = np.asarray(anomaly_scores)

    metrics = {
        "pr_auc": float(average_precision_score(y_true, anomaly_scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_anomaly_rate": float(np.mean(y_pred)),
    }

    # Ranking-based evaluation for fixed alert budgets.
    k = max(1, int(len(y_true) * top_k_ratio))
    top_k_idx = np.argsort(anomaly_scores)[-k:]
    top_k_true = y_true[top_k_idx]
    total_positives = int(np.sum(y_true))

    metrics["k_ratio"] = float(top_k_ratio)
    metrics["precision_at_k"] = float(np.mean(top_k_true))
    metrics["recall_at_k"] = (
        float(np.sum(top_k_true) / total_positives) if total_positives > 0 else 0.0
    )

    # Store full PR curve points for visualization.
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, anomaly_scores)
    metrics["pr_curve_precision"] = pr_precision.tolist()
    metrics["pr_curve_recall"] = pr_recall.tolist()

    return metrics


def _metrics_for_display(metrics):
    """Omit large PR-curve arrays from console output."""
    skip = {"pr_curve_precision", "pr_curve_recall"}
    return {k: v for k, v in metrics.items() if k not in skip}


# --------------------------------------------------------------------------------------------------------
# Pipeline: Transductive Approach C
# --------------------------------------------------------------------------------------------------------

def pipeline(msg_callback=noop_callback, report_callback=noop_callback, verb=VERBOSE):
    """
    Transductive Approach C pipeline:

    1. Load, clean, preprocess the FULL dataset
    2. Fit Isolation Forest on ALL data (transductive — no splitting)
    3. Generate anomaly scores for every sample
    4. Use ground-truth labels to find optimal threshold (F1-maximizing)
    5. Apply threshold -> binary predictions
    6. Evaluate with imbalance-aware metrics

    Parameters
    ----------
    msg_callback : callable
        Handles status messages (e.g. for Streamlit app).
    report_callback : callable
        Handles result report as dict.
    verb : bool
        Whether to print verbose messages.
    """

    msg = "\n\nStarting Transductive Approach C pipeline...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # --- STEP 1: DATA HANDLING ---

    msg = "\n\nLoading and preprocessing data...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    data, target = load_data()
    cleaned_data = clean_data(data)
    X = preprocess_data(cleaned_data)

    msg = f"\n\nData loaded: {X.shape[0]} samples, {X.shape[1]} features.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    n_anomalies = int(target.sum())
    anomaly_rate = float(target.mean())
    msg = f"Ground-truth: {n_anomalies} anomalies ({anomaly_rate:.4%} anomaly rate).\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # --- STEP 2: FIT ISOLATION FOREST ON ALL DATA (TRANSDUCTIVE) ---

    msg = "\n\nFitting Isolation Forest on the FULL dataset (transductive)...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    model = modeling(X)

    msg = "\n\nModel fitted.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # --- STEP 3: SCORE ALL SAMPLES ---

    msg = "\n\nGenerating anomaly scores for all samples...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    anomaly_scores = score_samples(model, X)

    msg = f"\n\nScores generated. Range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # --- STEP 4: FIND OPTIMAL THRESHOLD USING LABELS ---

    msg = "\n\nFinding optimal threshold using labeled data (F1-maximizing)...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    best_threshold, threshold_info = find_optimal_threshold(anomaly_scores, target)

    msg = (
        f"\n\nOptimal threshold found: {best_threshold:.6f}\n"
        f"  -> F1:        {threshold_info['best_f1']:.4f}\n"
        f"  -> Precision: {threshold_info['best_precision']:.4f}\n"
        f"  -> Recall:    {threshold_info['best_recall']:.4f}\n"
        f"  (evaluated {threshold_info['n_thresholds_evaluated']} candidate thresholds)\n"
    )
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # --- STEP 5: APPLY THRESHOLD ---

    msg = "\n\nApplying threshold to generate binary predictions...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    predictions = predict_with_threshold(anomaly_scores, best_threshold)

    n_predicted_anomalies = int(predictions.sum())
    msg = f"\n\nPredicted {n_predicted_anomalies} anomalies out of {len(predictions)} samples ({n_predicted_anomalies/len(predictions):.4%}).\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # --- STEP 6: EVALUATE ---

    msg = "\n\nEvaluating results...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    metrics = evaluate(target, predictions, anomaly_scores)

    print("\n" + "=" * 60)
    print("TRANSDUCTIVE APPROACH C — EVALUATION METRICS")
    print("=" * 60)
    for metric_name, metric_value in _metrics_for_display(metrics).items():
        print(f"  {metric_name}: {metric_value}")
    print("=" * 60)




    # --- REPORT ---

    to_report = {
        "predictions": predictions,
        "anomaly_scores": anomaly_scores,
        "metrics": metrics,
        "threshold_info": threshold_info,
    }

    report_callback(news=to_report)

    msg = "\n\nTransductive Approach C pipeline completed.\n\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg, end=True)

    return to_report


if __name__ == "__main__":
    pipeline(verb=True)
