# --------------------------------------------------------------------------------------------------------
# IMPORTANT
# --------------------------------------------------------------------------------------------------------
#
# This is the main development file.
# Please make sure you only add justified code here, otherwise your code may be rejected by the reviewers.
# For exploratory and experimental code, please use the `offline-detection/tasks` folder.
#
# --------------------------------------------------------------------------------------------------------

# Imports and constants
from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42
VERBOSE = True

# 1. Load data

def load_data():
    # TODO: Load the dataset and detach target variable from the data.
    # Output: data, target
    data, target = None, None
    return data, target

# 2.1 Cleaning

def clean_data(data):
    # TODO: include any cleaning steps that need to be done before preprocessing.
    # Output: cleaned_data
    cleaned_data = None
    return cleaned_data

# 2.2 Preprocessing

def preprocess_data(data):
    # TODO: Preprocess the data and convert into NumPy matrix format.
    # steps: cleaning, encoding, scaling, feature dropping, etc.
    # Output: X
    X = None
    return X


# 3. Modeling

def modeling(X):
    # TODO: Fit the Isolation Forest algorithm on the preprocessed data.
    # TODO: Settle on the initial hyperparameters of the model
    # Output: model
    hyparams = {"random_state": RANDOM_STATE, "verbose": VERBOSE}
    model = IsolationForest(**hyparams)
    return model


# 4. Prediction

def predict(model, X):
    # TODO: Use the fitted model to make predictions on the same data that it was fitted on.
    # Output: y_pred
    y_pred = None
    return y_pred


# 5. Evaluation

def evaluate(y_pred, target):
    # TODO: Evaluate the performance of the model using appropriate metrics.
    # Output: metrics
    metrics = None
    return metrics


# Actual pipeline execution

def pipeline():
    if VERBOSE: print("Starting pipeline execution...")
    # TODO: call methods in order ...
    if VERBOSE: print("Pipeline execution completed.")

pipeline()
