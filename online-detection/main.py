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

from river import datasets, metrics, compose, preprocessing, anomaly

def pipeline(
        process_n_observations=None, # Set to an integer to process only the first N samples, or None to process the entire dataset.
        report_every_seconds_elapsed=5*60,
        msg_callback=noop_callback,
        report_callback=noop_callback,
        random_seed=42,
        verb=True):

    msg = "\n\nStarting online anomaly detection on the Credit Card dataset...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # Load dataset
    
    dataset = datasets.CreditCard()

    if process_n_observations is None:
        process_n_observations = dataset.n_samples


    msg =f"\n\nIs the dataset downloaded? {dataset.is_downloaded}\n"
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


    msg = "\n\nPipeline initialized with MinMaxScaler and HalfSpaceTrees.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # Process the dataset and measure the performance


    msg = "\n\nStarting to process the dataset and measure performance...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    msg = f"\n\nProcessing only the first {process_n_observations} samples...\n" if process_n_observations < dataset.n_samples else "\n\nProcessing the entire dataset. This may take a while...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    to_report = {
        "meta": {
            "n_samples": dataset.n_samples
        },
        "step": 0,
        "data": [],
        "scores": [],
    }

    def clear_report(step=True):
        to_report["data"] = []
        to_report["scores"] = []
        if step: to_report["step"] += 1

    for i, (x, y) in enumerate(dataset.take(process_n_observations)):
        score = model.score_one(x)
        is_anomaly = model['QuantileFilter'].classify(score)

        model.learn_one(x)

        auc.update(y, score)
        class_report.update(y, is_anomaly)

        seconds_elapsed = x["Time"]

        to_report["data"].append(x)
        to_report["scores"].append({"score": score})

        if seconds_elapsed >= ((to_report["step"] + 1) * report_every_seconds_elapsed) or (i == process_n_observations-1):
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
