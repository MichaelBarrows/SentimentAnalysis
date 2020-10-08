import helpers
import dataset as ds
import pandas as pd
import run

# get_results_filenames()
# parameters:
#   version_path : string - the file path where the results files are stored
# returns
#   files : list - a list containing all of the results filenames
# description:
#   This function calls the path-fetcher function in the helpers file to
#       retrieve the filenames of the results files within that path. These
#       file names are then returned
def get_results_filenames (version_path):
    files = helpers.path_fetcher(version_path)
    return files

# get_first_experimental_results()
# parameters:
#   data : DataFrame - the data containing the data for processing
#   mpt : int - the match percentage threshold to be retained of the dataset
#   results_df : DataFrame - The dataframe holding the existing results
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function creates the first set of experimental results for each of the
#       four machine learning algorithms with hyperparameters by removing part
#       of the dataset with a words_matched_percentage value lower than the mpt
#       variable. The results are then stored in a list, converted (and
#       appended) to a dataframe and stored in a CSV file.
def get_first_experimental_results (data, mpt, results_df, n_grams):
    print("--- " + str(mpt) + "% ---")
    data = data[data.words_matched_percentage >= mpt]
    results = []

    results.append(run.run_knn_classification(data, 3, "initial experiment", n_grams))
    results.append(run.run_knn_classification(data, 7, "initial experiment", n_grams))

    results.append(run.run_linear_svm_classification(data, 0.8, "initial experiment", n_grams))
    results.append(run.run_linear_svm_classification(data, 1.2, "initial experiment", n_grams))

    results.append(run.run_naive_bayes_classification(data, 0.8, "initial experiment", n_grams))
    results.append(run.run_naive_bayes_classification(data, 1.2, "initial experiment", n_grams))

    results.append(run.run_random_forest_classification(data, [50, 100], "initial experiment", n_grams))
    results.append(run.run_random_forest_classification(data, [75, 150], "initial experiment", n_grams))
    columns = ["algorithm",
            "hyperparameter",
            "weighted_avg_precision",
            "weighted_avg_recall",
            "weighted_avg_f1-score",
            "accuracy",
            "experiment_type",
            "metric_dump_id",
            "positive_precision",
            "positive_recall",
            "positive_f1-score",
            "neutral_precision",
            "neutral_recall",
            "neutral_f1-score",
            "negative_precision",
            "negative_recall",
            "negative_f1-score"]
    results = pd.DataFrame(results, columns=columns)
    results_df = results_df.append(results)

    print(results_df)
    return results_df

# negation_handled()
# parameters:
#   folder : string - the output folder
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function get the filenames of existing results files and iterates over
#       the list. The Twitter data is loaded and for each results file
#       (different MPT value), the results file is loaded to a dataframe and
#       get_first_experimental_results() is called to generate the results.
#       the new results are stored.
def negation_handled (folder, n_grams):
    data = helpers.load_dataset(ds.dataset + ds.negate_dataset)
    results_files = get_results_filenames(folder)
    for results_file in results_files:
        mpt = results_file.split("_")[0]
        if mpt == "best":
            continue
        mpt = int(mpt)
        results_df = helpers.load_dataset(folder + results_file)
        results_df = get_first_experimental_results(data, mpt, results_df, n_grams)
        helpers.dataframe_to_csv(results_df, folder + results_file)


# negation_not_handled()
# parameters:
#   folder : string - the output folder
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function get the filenames of existing results files and iterates over
#       the list. The Twitter data is loaded and for each results file
#       (different MPT value), the results file is loaded to a dataframe and
#       get_first_experimental_results() is called to generate the results.
#       the new results are stored.
def negation_not_handled (folder, n_grams):
    results_files = get_results_filenames(folder)
    data = helpers.load_dataset(ds.dataset + ds.not_negate_dataset)
    for results_file in results_files:
        mpt = results_file.split("_")[0]
        if mpt == "best":
            continue
        mpt = int(mpt)
        results_df = helpers.load_dataset(folder + results_file)
        results_df = get_first_experimental_results(data, mpt, results_df, n_grams)
        helpers.dataframe_to_csv(results_df, folder + results_file)
