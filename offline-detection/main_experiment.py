# --------------------------------------------------------------------------------------------------------
# EXPERIMENT NOTES (IMBALANCED SCENARIO)
# --------------------------------------------------------------------------------------------------------
#
# Goal:
# - Evaluate anomaly detection in a highly imbalanced setting (rare positives).
#
# Key decisions made in this file:
# 1) Use Isolation Forest (instead of a dummy baseline) so anomalies can be detected.
# 2) Convert model output to 0/1 labels where 1 means anomaly, matching dataset target.
# 3) Use score-based metrics for imbalance:
#    - PR-AUC (primary ranking metric)
# 4) Keep threshold-based metrics for operations:
#    - Precision, Recall, F1, Predicted anomaly rate
# 5) Export anomaly scores + PR curve arrays for app/notebook visualization.
# 6) Temporal split uses a small validation slice (default 5%) for IF grid search (val PR-AUC);
#    final model is fit on train only with best params; test set is for reporting only.
#
# Why not accuracy:
# - In rare-event data, accuracy can be very high even when model misses nearly all anomalies.
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

from itertools import product

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

# Hold-out evaluation: temporal split (rows follow Time order in creditcard.csv).
# Small validation slice is used for IF hyperparameter search; test stays untouched until final metrics.
TEMPORAL_TRAIN_FRAC = 0.75
TEMPORAL_VAL_FRAC = 0.05
# Remaining fraction is test (0.20).

# Default IF settings when not tuning (or as fallback).
DEFAULT_IF_PARAMS = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": 0.002,
}

# Grid for validation PR-AUC search (imbalanced ranking metric).
IF_PARAM_GRID = {
    "contamination": (0.0005, 0.001, 0.002, 0.005, 0.01),
    "n_estimators": (100, 200, 300),
    "max_samples": ("auto", 256),
}


# 1. Load data

def load_data():
    """
    Step 1: Load the credit-card dataset and split:
    - data: all features
    - target: ground-truth anomaly labels (Class: 0 normal, 1 fraud/anomaly)
    """

    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_file = Path(dataset_path) / "creditcard.csv"
    df = pd.read_csv(csv_file)

    assert "Class" in df.columns, "Target variable 'Class' not found in the dataset."

    data = df.drop(columns=["Class"])
    target = df["Class"]

    assert "Class" not in data.columns, "Target variable 'Class' should not be part of the data."

    return data, target


# 2.1 Cleaning

def clean_data(data):
    """
    Step 2.1: Lightweight cleaning placeholder.
    Isolation Forest usually does not need heavy cleaning for this dataset.
    Keep this function so we can add missing-value handling or feature fixes later.
    """

    # TODO: ...

    cleaned_data = data.copy() #FIXME
    return cleaned_data


# 2.2 Preprocessing

def preprocess_data(data):
    """
    Step 2.2: Preprocessing for modeling.
    We scale only `Time` and `Amount` because PCA-like features (V1..V28)
    are already transformed in this dataset.

    For training+evaluation without leakage, use preprocess_fit / preprocess_transform instead.
    """

    df = data.copy()

    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    X = df.to_numpy()
    return X


def preprocess_fit(data_train):
    """Fit scaler on train only; return X_train and scaler for val/test transform."""
    df = data_train.copy()
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    return df.to_numpy(), scaler


def preprocess_transform(data, scaler):
    """Apply scaler fitted on training data (no leakage)."""
    df = data.copy()
    df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    return df.to_numpy()


def temporal_split_indices(n, train_frac=TEMPORAL_TRAIN_FRAC, val_frac=TEMPORAL_VAL_FRAC):
    """Chronological indices: train | validation | test."""
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    idx_train = np.arange(0, train_end)
    idx_val = np.arange(train_end, val_end)
    idx_test = np.arange(val_end, n)
    return idx_train, idx_val, idx_test


# 3. Modeling

def modeling(X, if_params=None):
    """
    Step 3: Fit Isolation Forest.

    if_params: optional dict overriding DEFAULT_IF_PARAMS (e.g. after validation tuning).

    Notes:
    - contamination should be in the ballpark of anomaly rate (here ~0.17% fraud).
    - Larger n_estimators usually stabilizes scores at higher fit cost.
    """
    params = {**DEFAULT_IF_PARAMS, "random_state": RANDOM_STATE, "n_jobs": -1}
    if if_params:
        params.update(if_params)
    model = IsolationForest(**params).fit(X)
    return model


def tune_isolation_forest_hyperparams(X_train, X_val, y_val, verb=True):
    """
    Choose IF hyperparameters by maximizing validation PR-AUC (ranking quality under imbalance).

    Fits only on X_train; scores are computed on X_val with labels y_val (no leakage into training).
    Returns (best_param_dict, best_pr_auc_on_val).
    """
    y_val = np.asarray(y_val).astype(int)
    best_pr_auc = -1.0
    best_params = None

    keys = tuple(IF_PARAM_GRID.keys())
    combos = list(product(*(IF_PARAM_GRID[k] for k in keys)))

    for combo in combos:
        cand = dict(zip(keys, combo))
        model = modeling(X_train, if_params=cand)
        _, scores_val = predict(model, X_val)
        pr_auc = average_precision_score(y_val, scores_val)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_params = cand.copy()

    if verb and best_params is not None:
        print(
            f"\n[Tuning] Best validation PR-AUC: {best_pr_auc:.6f}\n"
            f"[Tuning] Best IF params: {best_params}\n"
        )

    return best_params, float(best_pr_auc)


# 4. Prediction

def predict(model, X):
    """
    Step 4: Generate binary predictions and ranking scores.

    IsolationForest outputs:
    -  1 for inlier (normal)
    - -1 for outlier (anomaly)

    We map this to:
    - 0 for normal
    - 1 for anomaly

    We also return continuous anomaly scores for PR-AUC and top-k metrics.
    """

    raw_pred = model.predict(X)
    y_pred = (raw_pred == -1).astype(int)  # 1=anomaly, 0=normal
    anomaly_scores = -model.score_samples(X)  # higher means "more anomalous"
    return y_pred, anomaly_scores


# 5. Evaluation

def evaluate(target, y_pred, anomaly_scores, top_k_ratio=0.01):
    """
    Step 5: Evaluate with metrics appropriate for class imbalance.

    Reported metrics:
    - pr_auc: primary metric for rare-event ranking quality
    - precision / recall / f1: threshold-dependent quality at current model cutoff
    - predicted_anomaly_rate: operational signal (alert volume)
    - precision_at_k / recall_at_k: quality under a fixed review budget
      (top `k_ratio` highest anomaly scores)
    - pr_curve_precision / pr_curve_recall: arrays for plotting PR curve in app/notebook
    """

    y_true = np.asarray(target).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    anomaly_scores = np.asarray(anomaly_scores)

    # Threshold-dependent metrics at current IF decision boundary.
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

    # Store full PR curve points to visualize precision/recall trade-off.
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, anomaly_scores)
    metrics["pr_curve_precision"] = pr_precision.tolist()
    metrics["pr_curve_recall"] = pr_recall.tolist()

    return metrics


def _metrics_for_display(metrics):
    """Omit large PR-curve arrays from console output."""
    skip = {"pr_curve_precision", "pr_curve_recall"}
    return {k: v for k, v in metrics.items() if k not in skip}


# Metrics to compare when reporting baseline vs tuned (ranking + operational).
COMPARE_METRIC_KEYS = (
    "pr_auc",
    "precision_at_k",
    "recall_at_k",
    "f1",
    "precision",
    "recall",
)


def _load_holdout_splits(
    train_frac=TEMPORAL_TRAIN_FRAC,
    val_frac=TEMPORAL_VAL_FRAC,
):
    """
    Load data, chronological split, fit scaler on train only.
    Returns X_train, X_val, X_test, y_train, y_val, y_test, split_info.
    """
    data, target = load_data()
    cleaned = clean_data(data)
    n = len(cleaned)
    idx_train, idx_val, idx_test = temporal_split_indices(n, train_frac, val_frac)

    data_train = cleaned.iloc[idx_train]
    data_val = cleaned.iloc[idx_val]
    data_test = cleaned.iloc[idx_test]
    y_train = target.iloc[idx_train].to_numpy()
    y_val = target.iloc[idx_val].to_numpy()
    y_test = target.iloc[idx_test].to_numpy()

    X_train, scaler = preprocess_fit(data_train)
    X_val = preprocess_transform(data_val, scaler)
    X_test = preprocess_transform(data_test, scaler)

    split_info = {
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val": float(y_val.mean()),
        "positive_rate_test": float(y_test.mean()),
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, split_info


def _pct_delta(before: float, after: float) -> float:
    if before == 0.0:
        return float("nan") if after == 0.0 else float("inf")
    return (after - before) / abs(before) * 100.0


def compare_baseline_vs_tuned(
    msg_callback=noop_callback,
    report_callback=noop_callback,
    verb=VERBOSE,
    train_frac=TEMPORAL_TRAIN_FRAC,
    val_frac=TEMPORAL_VAL_FRAC,
    top_k_ratio=0.01,
):
    """
    Re-evaluate: same hold-out split, default IF vs grid-tuned IF.

    Trains baseline (DEFAULT_IF_PARAMS only), then runs validation PR-AUC tuning and retrains.
    Prints a side-by-side table; use **test** PR-AUC as the primary "is it better?" signal.
    """
    msg = "\n\nRe-evaluation: baseline vs tuned Isolation Forest (same split)...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    X_train, X_val, X_test, y_train, y_val, y_test, split_info = _load_holdout_splits(
        train_frac, val_frac
    )

    msg = (
        f"\nSplit — train: {split_info['n_train']}, val: {split_info['n_val']}, "
        f"test: {split_info['n_test']}\n"
        f"Positive rate — train: {split_info['positive_rate_train']:.4f}, "
        f"val: {split_info['positive_rate_val']:.4f}, test: {split_info['positive_rate_test']:.4f}\n"
    )
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    # --- Baseline (no hyperparameter search) ---
    if verb:
        print("\n[1/2] Baseline: DEFAULT_IF_PARAMS only.\n")
    model_base = modeling(X_train, if_params=None)
    pred_v_b, sc_v_b = predict(model_base, X_val)
    pred_t_b, sc_t_b = predict(model_base, X_test)
    m_val_b = evaluate(y_val, pred_v_b, sc_v_b, top_k_ratio=top_k_ratio)
    m_test_b = evaluate(y_test, pred_t_b, sc_t_b, top_k_ratio=top_k_ratio)

    # --- Tuned on validation PR-AUC, then one final fit on train ---
    if verb:
        print("\n[2/2] Tuned: grid search on validation PR-AUC, refit on train.\n")
    best_params, tuning_val_pr_auc = tune_isolation_forest_hyperparams(
        X_train, X_val, y_val, verb=verb
    )
    model_tuned = modeling(X_train, if_params=best_params)
    pred_v_t, sc_v_t = predict(model_tuned, X_val)
    pred_t_t, sc_t_t = predict(model_tuned, X_test)
    m_val_t = evaluate(y_val, pred_v_t, sc_v_t, top_k_ratio=top_k_ratio)
    m_test_t = evaluate(y_test, pred_t_t, sc_t_t, top_k_ratio=top_k_ratio)

    # Comparison table (validation + test).
    print("\n" + "=" * 72)
    print("BASELINE vs TUNED — key metrics (imbalanced)")
    print("=" * 72)
    print(f"{'Metric':<22} {'Split':<6} {'Baseline':>12} {'Tuned':>12} {'Δ%':>10}")
    print("-" * 72)
    for key in COMPARE_METRIC_KEYS:
        for split_name, mb, mt in (
            ("val", m_val_b, m_val_t),
            ("test", m_test_b, m_test_t),
        ):
            b, t = mb[key], mt[key]
            delta = _pct_delta(b, t)
            if np.isnan(delta):
                d_str = "    —    "
            elif np.isinf(delta):
                d_str = "   (+)   "
            else:
                d_str = f"{delta:+.1f}%"
            print(f"{key:<22} {split_name:<6} {b:12.6f} {t:12.6f} {d_str:>10}")
    print("=" * 72)

    test_pr_base = m_test_b["pr_auc"]
    test_pr_tuned = m_test_t["pr_auc"]
    if test_pr_tuned > test_pr_base:
        verdict = (
            f"Tuned model is BETTER on held-out test PR-AUC "
            f"({test_pr_tuned:.6f} vs {test_pr_base:.6f})."
        )
    elif test_pr_tuned < test_pr_base:
        verdict = (
            f"Baseline is better on held-out test PR-AUC "
            f"({test_pr_base:.6f} vs {test_pr_tuned:.6f}); consider a smaller grid or different split."
        )
    else:
        verdict = "Test PR-AUC unchanged between baseline and tuned."

    print(f"\nVerdict (primary: test PR-AUC): {verdict}\n")
    print(f"Best tuned params: {best_params}  (grid best val PR-AUC during search: {tuning_val_pr_auc:.6f})\n")

    to_report = {
        "split": split_info,
        "baseline": {
            "if_params": dict(DEFAULT_IF_PARAMS),
            "metrics_validation": m_val_b,
            "metrics_test": m_test_b,
        },
        "tuned": {
            "if_params": best_params,
            "tuning_val_pr_auc_from_grid": tuning_val_pr_auc,
            "metrics_validation": m_val_t,
            "metrics_test": m_test_t,
        },
        "verdict_test_pr_auc_improved": test_pr_tuned > test_pr_base,
    }

    report_callback(news=to_report)

    msg = "\nRe-evaluation complete.\n\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg, end=True)

    return to_report


def evaluate_pipeline_performance(
    msg_callback=noop_callback,
    report_callback=noop_callback,
    verb=VERBOSE,
    train_frac=TEMPORAL_TRAIN_FRAC,
    val_frac=TEMPORAL_VAL_FRAC,
    top_k_ratio=0.01,
    tune_hyperparameters=True,
    if_params_override=None,
):
    """
    Evaluate pipeline performance on held-out data (honest metrics).

    Steps:
    1. Load and clean data; split indices by time order (train | val | test).
    2. Fit StandardScaler on train only; transform val and test (no leakage).
    3. Optionally tune Isolation Forest on the small validation split (maximize val PR-AUC).
    4. Fit final Isolation Forest on train with chosen hyperparameters (no test leakage).
    5. Predict on validation and test; compute imbalance metrics on each split.

    tune_hyperparameters: if True, search IF_PARAM_GRID using validation PR-AUC.
    if_params_override: if set, skip grid search and use this dict (e.g. {"contamination": 0.002}).

    Returns a dict with keys: metrics_validation, metrics_test, best_if_params, and arrays for app use.
    """
    msg = "\n\nPipeline performance evaluation (temporal hold-out)...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    X_train, X_val, X_test, y_train, y_val, y_test, split_info = _load_holdout_splits(
        train_frac, val_frac
    )

    msg = (
        f"\nSplit sizes — train: {split_info['n_train']}, val: {split_info['n_val']}, "
        f"test: {split_info['n_test']}\n"
        f"Positive rate — train: {split_info['positive_rate_train']:.4f}, "
        f"val: {split_info['positive_rate_val']:.4f}, test: {split_info['positive_rate_test']:.4f}\n"
    )
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    best_if_params = None
    tuning_val_pr_auc = None

    if if_params_override is not None:
        best_if_params = dict(if_params_override)
        if verb:
            print(f"\n[Using fixed IF params] {best_if_params}\n")
    elif tune_hyperparameters:
        msg = "\n\nTuning IF hyperparameters on validation split (PR-AUC)...\n"
        verb_aware_print(msg, verb)
        msg_callback(msg=msg)
        best_if_params, tuning_val_pr_auc = tune_isolation_forest_hyperparams(
            X_train, X_val, y_val, verb=verb
        )
    else:
        best_if_params = None
        if verb:
            print("\n[Skipping tuning] Using DEFAULT_IF_PARAMS.\n")

    msg = "\n\nFitting final model on training split...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    model = modeling(X_train, if_params=best_if_params)

    pred_val, scores_val = predict(model, X_val)
    pred_test, scores_test = predict(model, X_test)

    metrics_val = evaluate(y_val, pred_val, scores_val, top_k_ratio=top_k_ratio)
    metrics_test = evaluate(y_test, pred_test, scores_test, top_k_ratio=top_k_ratio)

    print("\n--- Validation set (imbalanced metrics) ---")
    for name, val in _metrics_for_display(metrics_val).items():
        print(f"  {name}: {val}")

    print("\n--- Test set (imbalanced metrics) ---")
    for name, val in _metrics_for_display(metrics_test).items():
        print(f"  {name}: {val}")

    to_report = {
        "split": split_info,
        "best_if_params": best_if_params if best_if_params is not None else dict(DEFAULT_IF_PARAMS),
        "tuning_val_pr_auc": tuning_val_pr_auc,
        "metrics_validation": metrics_val,
        "metrics_test": metrics_test,
        "predictions_validation": pred_val,
        "anomaly_scores_validation": scores_val,
        "target_validation": y_val,
        "predictions_test": pred_test,
        "anomaly_scores_test": scores_test,
        "target_test": y_test,
    }

    report_callback(news=to_report)

    msg = "\n\nHold-out evaluation finished.\n\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg, end=True)

    return to_report


def pipeline(msg_callback=noop_callback, report_callback=noop_callback, verb=VERBOSE):
    """
    End-to-end execution:
    1) load/clean/preprocess
    2) train Isolation Forest
    3) predict labels + anomaly scores
    4) compute imbalance-aware metrics
    5) send results to callback for app/notebook usage

    Parameters
    ----------

    msg_callback : callable
        A function that handles messages at the call site. It should accept a `msg` argument for the message string and an optional `end` argument to indicate if it's the end of the pipeline.

    report_callback : callable
        A function that handles a report in form of a dictionary. It should accept a `news` argument for the report.

    verb : bool
        Whether to print verbose messages.
    """

    msg = "\n\nStarting pipeline execution...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # DATA HANDLING

    msg = "\n\nPreprocessing...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    data, target = load_data()
    cleaned_data = clean_data(data)
    X = preprocess_data(cleaned_data)

    msg = "\n\nFinished preprocessing.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    # MODELING

    msg = "\n\nObtaining model...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    model = modeling(X)

    msg = "\n\nFinished modeling.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    # PREDICTION

    msg = "\n\nPredicting...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    predictions, anomaly_scores = predict(model, X)

    msg = "\n\nPredictions created.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    # EVALUATION

    msg = "\n\nEvaluating results...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    metrics = evaluate(target, predictions, anomaly_scores)

    print("\nEvaluation Metrics (full data — optimistic; use evaluate_pipeline_performance for hold-out):")
    for metric_name, metric_value in _metrics_for_display(metrics).items():
        print(f"{metric_name}: {metric_value}")

    # Send all artifacts needed for demo plots/tables.
    to_report = {
        "predictions": predictions,
        "anomaly_scores": anomaly_scores,
        "metrics": metrics,
    }

    report_callback(news=to_report)

    msg = "\n\nFinished evaluation.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)
    
    msg = "\n\nPipeline execution completed.\n\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg, end=True)


if __name__ == "__main__":
    # Re-evaluate baseline vs tuned on the same split; verdict uses test PR-AUC.
    compare_baseline_vs_tuned(verb=True)
