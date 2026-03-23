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
        verb=True):

    msg = "\n\nStarting online anomaly detection on the Credit Card dataset...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # Load dataset
    dataset = datasets.CreditCard()


    msg =f"\n\nIs the dataset downloaded? {dataset.is_downloaded}\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # Evaluation metrics
    auc = metrics.ROCAUC()


    msg ="\n\nUsing metrics of ROC-AUC to measure performance.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # Model pipeline
    model = compose.Pipeline(
        preprocessing.MinMaxScaler(),
        anomaly.HalfSpaceTrees()
    )


    msg = "\n\nPipeline initialized with MinMaxScaler and HalfSpaceTrees.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)


    # Process the dataset and measure the performance


    msg = "\n\nStarting to process the dataset and measure performance...\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)

    msg = f"\n\nProcessing only the first {process_n_observations} samples...\n" if process_n_observations is not None else "\n\nProcessing the entire dataset. This may take a while...\n"
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
        model.learn_one(x)
        auc.update(y, score)

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

    msg = "\n\nFinished evaluation.\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg)
    
    msg = "\n\nPipeline execution completed.\n\n"
    verb_aware_print(msg, verb)
    msg_callback(msg=msg, end=True)
