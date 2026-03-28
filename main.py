# --------------------------------------------------------------------------------------------------------
# IMPORTANT
# --------------------------------------------------------------------------------------------------------
#
# This is the main development file.
# Please make sure you only add justified code here, otherwise your code may be rejected by the reviewers.
# For exploratory and experimental code, please use the `online-detection/tasks` folder.
#
# --------------------------------------------------------------------------------------------------------

from river import datasets, metrics, compose, preprocessing, anomaly
import numpy as np

VERBOSE = True

# Set to an integer to process only the first N samples, or None to process the entire dataset.
PROCESS_N_OBSERVATIONS = None

if VERBOSE: print("\nStarting online anomaly detection on the Credit Card dataset...\n")

# Load dataset

dataset = datasets.CreditCard()
if VERBOSE: print("Is the dataset downloaded? ", dataset.is_downloaded, "\n")

# Evaluation metrics

auc = metrics.ROCAUC()
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1()
if VERBOSE: print("Using metrics of ROC-AUC to measure performance.\n")

scores_history = []
WINDOW_SIZE = 1000
# Model pipeline

model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.HalfSpaceTrees()
)

for i, (x, y) in enumerate(dataset):

    # Gets the anomaly score
    score = model.score_one(x)

    # Stores the scores
    scores_history.append(score)
    if len(scores_history) > WINDOW_SIZE:
        scores_history.pop(0)

    # Compute threshold after enough data
    if len(scores_history) > 50:
        threshold = np.percentile(scores_history, 99)
    else:
        threshold = 0.5  # fallback at the start

    # Convert to binary prediction
    y_pred = 1 if score >= threshold else 0

    # Update metrics
    auc.update(y, score)
    precision.update(y, y_pred)
    recall.update(y, y_pred)
    f1.update(y, y_pred)

    # Train model
    model.learn_one(x)

print("\nFinal Results:\n")
print("ROC-AUC:", auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)