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

def verb_aware_print(msg, verb=True):
    if verb and msg: print(msg)

def noop_callback(**kwargs):
    pass

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

def pipeline(
        dataset,
        threshold_tuning_params: ColdStartThresholdTuningParams = None,
        report_every_seconds_elapsed=5*60,
        msg_callback=noop_callback,
        report_callback=noop_callback,
        random_seed=42,
        verb=True):
    
    tuned_initial_threshold = None

    msg = "\n\nStarting online anomaly detection on the Credit Card dataset...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    # Evaluation metrics

    ## thresholdless
    auc = metrics.ROCAUC()

    ## threshold-based
    class_report = metrics.ClassificationReport()


    msg ="\n\nUsing metrics of ROC-AUC to measure performance.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # Model pipeline
    model = compose.Pipeline(
        preprocessing.MinMaxScaler(),
        anomaly.QuantileFilter(
            anomaly.HalfSpaceTrees(seed=random_seed),
            q=.99
        )
    )

    # Initial threshold
    if threshold_tuning_params is not None:
        msg = "\n\nThreshold tuning parameters provided. Starting cold start threshold tuning...\n"
        verb_aware_print(msg, verb)
        msg_callback(msg=msg)

        scores_for_tuning = []
        y_true_for_tuning = []

        for x, y in threshold_tuning_params.labeled_data:
            score = model.score_one(x)
            scores_for_tuning.append(score)
            y_true_for_tuning.append(y)

            model.learn_one(x)
        
        if len(set(y_true_for_tuning)) < 2:
            msg = "\n\nNot enough class diversity in the labeled data for threshold tuning. Skipping tuning.\n"
            verb_aware_print(msg, verb)
            msg_callback(msg=msg)
        else:
            tuned_initial_threshold, _ = finetune_threshold_f1(y_true_for_tuning, scores_for_tuning)

            msg = f"\n\nCold start threshold tuning completed. Threshold is set to: {tuned_initial_threshold:.4f}\n"
            verb_aware_print(msg, verb)
            msg_callback(msg=msg)


    msg = "\n\nPipeline initialized with MinMaxScaler and HalfSpaceTrees.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # Process the dataset and measure the performance


    msg = "\n\nStarting to process the dataset and measure performance...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    to_report = {
        "step": 0,
        "data": [],
        "scores": [],
    }

    def clear_report():
        to_report["data"] = []
        to_report["scores"] = []
        to_report["step"] += 1

    # FIXME: model may learn from validation data but shall not be included in the evaluation!

    for i, (x, y_for_evaluation_only) in enumerate(dataset):
        seconds_elapsed = x["Time"]
        use_tuned_threshold = tuned_initial_threshold is not None and seconds_elapsed <= threshold_tuning_params.invalidate_after_seconds

        score = model.score_one(x)
        is_anomaly = model['QuantileFilter'].classify(score) if not use_tuned_threshold else (score >= tuned_initial_threshold)

        model.learn_one(x)

        auc.update(y_for_evaluation_only, score)
        class_report.update(y_for_evaluation_only, is_anomaly)

        to_report["data"].append(x)
        to_report["scores"].append({"score": score})

        if seconds_elapsed >= ((to_report["step"] + 1) * report_every_seconds_elapsed):
            report_callback(news=to_report)
            clear_report()
        
        if tuned_initial_threshold is not None and seconds_elapsed > threshold_tuning_params.invalidate_after_seconds:
            tuned_initial_threshold = None
            msg = f"\n\nInvalidating the tuned threshold at {seconds_elapsed} seconds. Switching to QuantileFilter.\n"
            verb_aware_print(msg, verb)
            msg_callback(msg=msg)

    if len(to_report["data"]) > 0:
        report_callback(news=to_report)
        clear_report()
            

    msg = "\n\nFinished processing the dataset.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # Final evaluation
    rocauc = auc.get()


    msg = f"\n\nFinal ROC-AUC: {rocauc:.4f}\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    msg = f"\n\nClassification Report:\n{class_report}\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    msg = "\n\nFinished evaluation.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)
    
    msg = "\n\nPipeline execution completed.\n\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg, end=True)
