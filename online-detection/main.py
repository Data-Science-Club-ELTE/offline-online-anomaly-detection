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

VERBOSE = True

# Set to an integer to process only the first N samples, or None to process the entire dataset.
PROCESS_N_OBSERVATIONS = None

if VERBOSE: print("\nStarting online anomaly detection on the Credit Card dataset...\n")


# Load dataset

dataset = datasets.CreditCard()
if VERBOSE: print("Is the dataset downloaded? ", dataset.is_downloaded, "\n")


# Evaluation metrics

auc = metrics.ROCAUC()
if VERBOSE: print("Using metrics of ROC-AUC to measure performance.\n")


# Model pipeline
    
model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.HalfSpaceTrees()
)
if VERBOSE: print("Pipeline initialized with MinMaxScaler and HalfSpaceTrees.\n")


# Process the dataset and measure the performance

if VERBOSE:
    print("Starting to process the dataset and measure performance...\n")
    if PROCESS_N_OBSERVATIONS is not None: print(f"Processing only the first {PROCESS_N_OBSERVATIONS} samples...\n")
    else: print("Processing the entire dataset. This may take a while...\n")

for x, y in dataset.take(PROCESS_N_OBSERVATIONS):
    score = model.score_one(x)
    model.learn_one(x)
    auc.update(y, score)

if VERBOSE: print("\nFinished processing the dataset.\n")


# Final evaluation

print(f"Final ROC-AUC: {auc.get():.4f}\n\n")
