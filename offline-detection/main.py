# --------------------------------------------------------------------------------------------------------
# IMPORTANT
# --------------------------------------------------------------------------------------------------------
#
# This is the main development file.
# Please make sure you only add justified code here, otherwise your code may be rejected by the reviewers.
# For exploratory and experimental code, please use the `offline-detection/tasks` folder.
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

RANDOM_STATE = 42
VERBOSE = True


# 1. Load data

def load_data():
    """
    Load the credit card fraud dataset and separate features from the target variable.
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
    Perform any necessary data cleaning steps on the raw data before preprocessing.
    Do consider our algorithm, Isolation Forest, may not require extensive cleaning.
    """

    # TODO: ...

    cleaned_data = data.copy() #FIXME
    return cleaned_data


# 2.2 Preprocessing

def preprocess_data(data):
    """
    Preprocess the cleaned data to make it suitable for modeling with Isolation Forest.
    Encode, scale, drop feature(s) if necessary, etc., convert into a numpy array.
    """

    # TODO: ...

    X = data.copy().to_numpy() #FIXME
    return X


# 3. Modeling

def modeling(X):
    """
    Select hyperparameters and fit the Isolation Forest algorithm on the preprocessed data.
    """

    # TODO remove dummy, use Isolation Forest
    from sklearn.dummy import DummyClassifier
    hyparams = {"random_state": RANDOM_STATE, "verbose": VERBOSE}
    model = DummyClassifier().fit(X, np.zeros(X.shape[0]))

    return model


# 4. Prediction

def predict(model, X):
    """
    Use the fitted model to make predictions on the same data that it was fitted on.
    """

    y_pred = model.predict(X) #FIXME
    return y_pred


# 5. Evaluation

def evaluate(target, y_pred):
    """
    Evaluate the performance of the model using appropriate metrics for anomaly detection.
    """

    # FIXME
    # Extremenly important! DummyClassifier gives 0.99 / 0.00 / 0.00 for accuracy / recall / precision, which is useless for anomaly detection.
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    metrics = {
        "accuracy": accuracy_score(target, y_pred),
        "recall": recall_score(target, y_pred),
        "precision": precision_score(target, y_pred)
    }

    return metrics


def pipeline(
        cached_dataset:tuple=None,
        msg_callback=noop_callback,
        report_callback=noop_callback,
        verb=VERBOSE):
    """
    Pipeline function that executes the entire workflow from data loading to evaluation, while providing updates through callbacks (e.g.: for the Streamlit app).

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

    data, target = cached_dataset or load_data()
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

    predictions = predict(model, X)

    msg = "\n\nPredictions created.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    # EVALUATION

    msg = "\n\nEvaluating results...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    metrics = evaluate(target, predictions)

    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    to_report = {
        "predictions": predictions,
        "metrics": metrics,
    }

    report_callback(news=to_report)

    msg = "\n\nFinished evaluation.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)
    
    msg = "\n\nPipeline execution completed.\n\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg, end=True)
