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

    validation_size = int(0.3 * process_n_observations)

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

    # Candidate models (for tuning)
    models = {
        "q_0.95": compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.QuantileFilter(
                anomaly.HalfSpaceTrees(seed=random_seed),
                q=0.95
            )
        ),
        "q_0.99": compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.QuantileFilter(
                anomaly.HalfSpaceTrees(seed=random_seed),
                q=0.99
            )
        ),
        "q_0.999": compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.QuantileFilter(
                anomaly.HalfSpaceTrees(seed=random_seed),
                q=0.999
            )
        )
    }

    # Track validation performance
    model_auc_scores = {name: metrics.ROCAUC() for name in models}
    best_model = None


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
        score = None
        if i < validation_size:
            # Validation

            for name, m in models.items():
                score = m.score_one(x)
                model_auc_scores[name].update(y, score)
                m.learn_one(x)

        else:
            # Testing Phase

            if best_model is None:
                # Select best model based on validation ROC-AUC
                best_model_name = max(model_auc_scores, key=lambda k: model_auc_scores[k].get())
                best_model = models[best_model_name]

                msg = f"\n\nSelected best model: {best_model_name}\n"
                verb_aware_print(msg, verb)
                msg_callback(msg=msg)

            score = best_model.score_one(x)
            is_anomaly = best_model['QuantileFilter'].classify(score)

            best_model.learn_one(x)

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

