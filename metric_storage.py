import helpers
import dataset as ds
import pandas as pd

# get_metric_storage_filenames()
# parameters:
#   None
# returns
#   files : list - a list containing all of the metric files
# description:
#   This function calls the path-fetcher function in the helpers file to
#       retrieve the filenames of the results files within that path. These
#       file names are then returned
def get_metric_storage_filenames ():
    files = helpers.path_fetcher(ds.metric_storage_location)
    return files

def get_new_metric_storage_identifier ():
    existing_files = get_metric_storage_filenames()
    ids = []
    if len(existing_files) == 0:
        metric_storage_id = 1
        return metric_storage_id

    for file in existing_files:
        ids.append(int(file.split(".")[0]))
    ids.sort()
    metric_storage_id = int(ids[-1]) + 1
    return metric_storage_id

def store_metrics (metrics_dict, algorithm, modifier, n_grams):
    location = ds.metric_storage_location
    metric_id = get_new_metric_storage_identifier()
    metric_list_for_df = []

    for index in range(1,11):
        metric_list_for_df.append([
            metric_id,
            index,
            algorithm,
            modifier,
            n_grams,
            metrics_dict["Positive"]["precision"][index - 1],
            metrics_dict["Positive"]["recall"][index - 1],
            metrics_dict["Positive"]["f1-score"][index - 1],
            metrics_dict["Positive"]["support"][index - 1],
            metrics_dict["Neutral"]["precision"][index - 1],
            metrics_dict["Neutral"]["recall"][index - 1],
            metrics_dict["Neutral"]["f1-score"][index - 1],
            metrics_dict["Neutral"]["support"][index - 1],
            metrics_dict["Negative"]["precision"][index - 1],
            metrics_dict["Negative"]["recall"][index - 1],
            metrics_dict["Negative"]["f1-score"][index - 1],
            metrics_dict["Negative"]["support"][index - 1],
            metrics_dict["accuracy"]["list"][index - 1],
            metrics_dict["macro avg"]["precision"][index - 1],
            metrics_dict["macro avg"]["recall"][index - 1],
            metrics_dict["macro avg"]["f1-score"][index - 1],
            metrics_dict["macro avg"]["support"][index - 1],
            metrics_dict["weighted avg"]["precision"][index - 1],
            metrics_dict["weighted avg"]["recall"][index - 1],
            metrics_dict["weighted avg"]["f1-score"][index - 1],
            metrics_dict["weighted avg"]["support"][index - 1],
        ])
    metric_list_for_df.append([
        metric_id,
        "average",
        algorithm,
        modifier,
        n_grams,
        metrics_dict["Positive"]["avg"]["precision"],
        metrics_dict["Positive"]["avg"]["recall"],
        metrics_dict["Positive"]["avg"]["f1-score"],
        metrics_dict["Positive"]["avg"]["support"],
        metrics_dict["Neutral"]["avg"]["precision"],
        metrics_dict["Neutral"]["avg"]["recall"],
        metrics_dict["Neutral"]["avg"]["f1-score"],
        metrics_dict["Neutral"]["avg"]["support"],
        metrics_dict["Negative"]["avg"]["precision"],
        metrics_dict["Negative"]["avg"]["recall"],
        metrics_dict["Negative"]["avg"]["f1-score"],
        metrics_dict["Negative"]["avg"]["support"],
        metrics_dict["accuracy"]["avg"],
        metrics_dict["macro avg"]["avg"]["precision"],
        metrics_dict["macro avg"]["avg"]["recall"],
        metrics_dict["macro avg"]["avg"]["f1-score"],
        metrics_dict["macro avg"]["avg"]["support"],
        metrics_dict["weighted avg"]["avg"]["precision"],
        metrics_dict["weighted avg"]["avg"]["recall"],
        metrics_dict["weighted avg"]["avg"]["f1-score"],
        metrics_dict["weighted avg"]["avg"]["support"],
    ])

    columns = ["metric_dump_id",
                "fold",
                "algorithm",
                "modifier",
                "n_grams",
                "positive_precision",
                "positive_recall",
                "positive_f1-score",
                "positive_support",
                "neutral_precision",
                "neutral_recall",
                "neutral_f1-score",
                "neutral_support",
                "negative_precision",
                "negative_recall",
                "negative_f1-score",
                "negative_support",
                "accuracy",
                "macro_avg_precision",
                "macro_avg_recall",
                "macro_avg_f1-score",
                "macro_avg_support",
                "weighted_avg_precision",
                "weighted_avg_recall",
                "weighted_avg_f1-score",
                "weighted_avg_support"]

    metric_df = pd.DataFrame(metric_list_for_df, columns=columns)
    helpers.dataframe_to_csv(metric_df, location + str(metric_id) + ".csv")
    return metric_id
