# --------------------------------------------------------------------------------------------------------
# IMPORTANT
# --------------------------------------------------------------------------------------------------------
#
# This is the main development file.
# Please make sure you only add justified code here, otherwise your code may be rejected by the reviewers.
# For exploratory and experimental code, please use the `offline-detection/tasks` folder.
#
# --------------------------------------------------------------------------------------------------------

import kagglehub
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Callable

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

DEFAULT_IF_PARAMS = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": "auto",
}

# --------------------------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------------------------

def verb_aware_print(msg: str, verb: bool = True):
    if verb and msg:
        print(msg)

def noop_callback(**kwargs):
    pass

def _log(msg: str, verb: bool, msg_callback: Callable, end: bool = False):
    """Helper to handle both console printing and UI callbacks."""
    verb_aware_print(msg, verb)
    msg_callback(msg=msg, end=end)

# --------------------------------------------------------------------------------------------------------
# Pipeline Components
# --------------------------------------------------------------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Retrieves credit card dataset and separates features from the target variable."""
    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_file = Path(dataset_path) / "creditcard.csv"
    df = pd.read_csv(csv_file)

    assert "Class" in df.columns, "Target variable 'Class' not found in dataset."

    initial_shape = df.shape
    df = df.drop_duplicates(keep="first")
    if initial_shape[0] - df.shape[0] > 0:
        print(f"\n[load_data] Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")

    data = df.drop(columns=["Class"])
    target = df["Class"]

    return data, target

def report_data_quality(df: pd.DataFrame, title: str) -> None:
    """Logs dataset shape and missing value statistics."""
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    print(f"\n===== {title} =====")
    print(f"Shape: {df.shape} | Duplicate rows: {df.duplicated().sum()} | Missing values: {missing.sum()}")
    if not missing_cols.empty:
        for col, count in missing_cols.items():
            print(f"  - {col}: {count}")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing numeric features via median and categorical via mode."""
    if df.isna().sum().sum() == 0:
        return df

    cleaned = df.copy()
    
    num_cols = cleaned.select_dtypes("number").columns
    cleaned[num_cols] = cleaned[num_cols].fillna(cleaned[num_cols].median())

    obj_cols = cleaned.select_dtypes(exclude="number").columns
    for col in obj_cols:
        if cleaned[col].isna().any():
            mode_val = cleaned[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
            cleaned[col] = cleaned[col].fillna(fill_val)

    return cleaned

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates data cleaning steps."""
    report_data_quality(data, "Before Cleaning")
    df = handle_missing_values(data)
    report_data_quality(df, "After Cleaning")
    return df

def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    """Scales numeric features and converts data to a numpy array."""
    df = data.copy()
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    return df.to_numpy()

def modeling(X: np.ndarray, if_params: dict = None) -> IsolationForest:
    """Fits an Isolation Forest model transductively on the complete dataset."""
    params = {**DEFAULT_IF_PARAMS, "random_state": RANDOM_STATE, "n_jobs": -1}
    if if_params:
        params.update(if_params)
    return IsolationForest(**params).fit(X)

def score_samples(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Generates continuous anomaly scores. Higher values indicate higher anomaly probability."""
    return -model.score_samples(X)

def find_optimal_threshold(anomaly_scores: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Calculates the threshold that maximizes the F1 score based on precision-recall curves."""
    labels = np.asarray(labels).astype(int)
    anomaly_scores = np.asarray(anomaly_scores)

    precisions, recalls, thresholds = precision_recall_curve(labels, anomaly_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])

    return best_threshold, {
        "best_f1": float(f1_scores[best_idx]),
        "best_precision": float(precisions[best_idx]),
        "best_recall": float(recalls[best_idx]),
        "threshold": best_threshold,
    }

def predict_with_threshold(anomaly_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Applies a scalar threshold to continuous scores to produce binary classifications."""
    return (anomaly_scores >= threshold).astype(int)

def evaluate(target: np.ndarray, y_pred: np.ndarray, anomaly_scores: np.ndarray, top_k_ratio: float = 0.01) -> Dict[str, float]:
    """Generates a suite of evaluation metrics standard for imbalanced anomaly detection."""
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

    k = max(1, int(len(y_true) * top_k_ratio))
    top_k_idx = np.argsort(anomaly_scores)[-k:]
    top_k_true = y_true[top_k_idx]
    total_positives = int(np.sum(y_true))

    metrics["k_ratio"] = float(top_k_ratio)
    metrics["precision_at_k"] = float(np.mean(top_k_true))
    metrics["recall_at_k"] = float(np.sum(top_k_true) / total_positives) if total_positives > 0 else 0.0

    pr_precision, pr_recall, _ = precision_recall_curve(y_true, anomaly_scores)
    metrics["pr_curve_precision"] = pr_precision.tolist()
    metrics["pr_curve_recall"] = pr_recall.tolist()

    return metrics

def _metrics_for_display(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Filters large array metrics from standard logging output."""
    return {k: v for k, v in metrics.items() if k not in {"pr_curve_precision", "pr_curve_recall"}}

# --------------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------------

def pipeline(msg_callback: Callable = noop_callback, report_callback: Callable = noop_callback, verb: bool = VERBOSE, tuning_fraction: float = 0.5) -> Dict[str, Any]:
    """
    Executes the end-to-end anomaly detection pipeline natively configured for transductive analysis.
    Uses a chronological split where the first `tuning_fraction` of the data calibrates the threshold, 
    and the remaining future data evaluates the final isolated F1 impact.
    """
    _log("\nStarting pipeline execution...", verb, msg_callback)

    _log("Loading and preprocessing data...", verb, msg_callback)
    data, target = load_data()
    cleaned_data = clean_data(data)
    X = preprocess_data(cleaned_data)

    n_anomalies = int(target.sum())
    _log(f"Data ready: {X.shape[0]} samples, {X.shape[1]} features. Anomalies: {n_anomalies} ({target.mean():.4%}).", verb, msg_callback)

    _log("Fitting Transductive Isolation Forest model...", verb, msg_callback)
    model = modeling(X)

    _log("Generating anomaly scores from all samples...", verb, msg_callback)
    anomaly_scores = score_samples(model, X)

    t_split = int(len(X) * tuning_fraction)
    _log(f"Applying chronological threshold split: {t_split} past / {len(X) - t_split} future", verb, msg_callback)

    # 1. TUNING PHASE (on historical data)
    _log("Optimizing decision boundary strictly on historical data...", verb, msg_callback)
    past_scores = anomaly_scores[:t_split]
    past_target = target.iloc[:t_split]
    best_threshold, threshold_info = find_optimal_threshold(past_scores, past_target)
    _log(f"Historical Optimal Threshold: {best_threshold:.6f} (Historical F1: {threshold_info['best_f1']:.4f})", verb, msg_callback)

    # 2. EVALUATION PHASE (on future, unseen data)
    _log("Generating binary predictions for future unseen data...", verb, msg_callback)
    future_scores = anomaly_scores[t_split:]
    future_target = target.iloc[t_split:]
    future_predictions = predict_with_threshold(future_scores, best_threshold)

    _log("Evaluating future validation results...", verb, msg_callback)
    metrics = evaluate(future_target, future_predictions, future_scores)

    if verb:
        print("\nFuture Unseen Evaluation Metrics:")
        for k, v in _metrics_for_display(metrics).items():
            print(f"  {k}: {v}")

    to_report = {
        "future_predictions": future_predictions,
        "future_anomaly_scores": future_scores,
        "metrics": metrics,
        "threshold_info": threshold_info,
    }
    report_callback(news=to_report)

    _log("\nPipeline execution completed.", verb, msg_callback, end=True)
    return to_report


if __name__ == "__main__":
    pipeline(verb=True)
