# --------------------------------------------------------------------------------------------------------
# IMPORTANT
# --------------------------------------------------------------------------------------------------------
#
# This is the main development file.
# Please make sure you only add justified code here, otherwise your code may be rejected by the reviewers.
# For exploratory and experimental code, please use the `online-detection/tasks` folder.
#
# --------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------
# Utilities

from typing import Callable

def verb_aware_print(msg, verb=True):
    if verb and msg: print(msg)

def noop_callback(**kwargs):
    pass

def _log(msg: str, verb: bool, msg_callback: Callable, end: bool = False):
    """Helper to handle both console printing and UI callbacks."""
    verb_aware_print(msg, verb)
    msg_callback(msg=msg, end=end)

# --------------------------------------------------------------------------------------------------------

import numpy as np

from typing import Literal
from dataclasses import dataclass
from river import datasets, metrics, compose, preprocessing, anomaly
from sklearn.metrics import precision_recall_curve

@dataclass
class ColdStartThresholdTuningParams:
    labeled_data: datasets.CreditCard
    optimize_for: Literal["f1"] = "f1"
    invalidate_after_seconds: int = 60*60

def finetune_threshold_f1(y_true, scores):
	_precision, _recall, _candidate_thresholds = precision_recall_curve(y_true, scores, pos_label=1)
	_f1_scores = 2 * (_precision * _recall) / (_precision + _recall)
	_f1_scores = np.nan_to_num(_f1_scores)
	_best_idx = np.argmax(_f1_scores)
	_best_score = _f1_scores[_best_idx]
	_best_threshold = _candidate_thresholds[_best_idx]
	return _best_threshold, _best_score

PERCENTILE = .99

def pipeline(
        dataset,
        process_n_observations=None,
        threshold_tuning_params: ColdStartThresholdTuningParams = None,
        report_every_seconds_elapsed=5*60,
        msg_callback=noop_callback,
        report_callback=noop_callback,
        random_seed=42,
        verb=True):
    
    tuned_initial_threshold = None

    _log("Starting online anomaly detection on the Credit Card dataset...", verb, msg_callback)

    # Evaluation metrics
    class_report = metrics.ClassificationReport()
    precision = metrics.Precision(pos_val=1)
    recall = metrics.Recall(pos_val=1)
    f1 = metrics.F1(pos_val=1)

    # Model pipeline
    model = compose.Pipeline(
        preprocessing.MinMaxScaler(),
        anomaly.QuantileFilter(
            anomaly.HalfSpaceTrees(seed=random_seed),
            q=PERCENTILE
        )
    )

    _log("Pipeline initialized with MinMaxScaler and HalfSpaceTrees.", verb, msg_callback)

    # Initial threshold
    if threshold_tuning_params is not None:
        _log("Threshold tuning parameters provided. Starting cold start threshold tuning...", verb, msg_callback)

        scores_for_tuning = []
        y_true_for_tuning = []

        for x, y in threshold_tuning_params.labeled_data:
            score = model.score_one(x)
            scores_for_tuning.append(score)
            y_true_for_tuning.append(y)
            model.learn_one(x)
        
        if len(set(y_true_for_tuning)) < 2:
            _log("Not enough classes in the labeled data for tuning. Skipping threshold tuning.", verb, msg_callback)

        else:
            tuned_initial_threshold, _ = finetune_threshold_f1(y_true_for_tuning, scores_for_tuning)
            _log(f"Cold start threshold tuning completed. Threshold is set to: {tuned_initial_threshold:.4f}", verb, msg_callback)

    _log("Starting to process the dataset and measure performance...", verb, msg_callback)

    to_report = {
        "step": 0,
        "data": [],
        "scores": [],
        "predictions": [],
    }

    def clear_report():
        to_report["data"] = []
        to_report["scores"] = []
        to_report["predictions"] = []
        to_report["step"] += 1

    T_0 = None
    COLD_BEFORE = None

    for i, (x, y_for_evaluation_only) in enumerate(dataset):
        if process_n_observations is not None and i >= process_n_observations:
            break

        seconds_elapsed = x["Time"]

        if i == 0:
            T_0 = seconds_elapsed
            COLD_BEFORE = T_0 + threshold_tuning_params.invalidate_after_seconds if threshold_tuning_params is not None else 0

        use_tuned_threshold = tuned_initial_threshold is not None and seconds_elapsed <= COLD_BEFORE

        score = model.score_one(x)
        is_anomaly = model['QuantileFilter'].classify(score) if not use_tuned_threshold else (score >= tuned_initial_threshold)

        model.learn_one(x)
        class_report.update(y_for_evaluation_only, is_anomaly)
        precision.update(y_for_evaluation_only, is_anomaly)
        recall.update(y_for_evaluation_only, is_anomaly)
        f1.update(y_for_evaluation_only, is_anomaly)

        to_report["data"].append(x)
        to_report["scores"].append({"score": score})
        to_report["predictions"].append({"prediction": is_anomaly})
        if seconds_elapsed >= ((to_report["step"] + 1) * report_every_seconds_elapsed):

            to_report["class_report"] = class_report
            to_report["precision"] = precision
            to_report["recall"] = recall
            to_report["f1"] = f1

            report_callback(news=to_report)
            clear_report()
        
        if tuned_initial_threshold is not None and seconds_elapsed > COLD_BEFORE:
            tuned_initial_threshold = None
            msg = f"\n\nInvalidating the tuned threshold at {seconds_elapsed} seconds. Switching to QuantileFilter with q={PERCENTILE}.\n"
            _log(msg, verb, msg_callback)


    if len(to_report["data"]) > 0:

        to_report["class_report"] = class_report
        to_report["precision"] = precision
        to_report["recall"] = recall
        to_report["f1"] = f1
        
        report_callback(news=to_report)
        clear_report()

    _log("Finished processing the dataset.", verb, msg_callback)
    _log(f"Classification Report:\n{class_report}", verb, msg_callback)
    _log("Finished evaluation.", verb, msg_callback)
    _log("Pipeline execution completed.", verb, msg_callback)
