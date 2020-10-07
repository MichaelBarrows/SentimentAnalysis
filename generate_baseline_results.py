import helpers
import dataset as ds
import pandas as pd
import run

# get_baseline_results()
# parameters:
#   data : DataFrame - the data containing the data for processing
#   mpt : int - the match percentage threshold to be retained of the dataset
#   output_folder : string - the folder path for storing CSV files
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function creates baseline results for each of the five machine learning
#       algorithms by removing part of the dataset with a
#       words_matched_percentage value lower than the mpt variable. The results
#       are then stored in a list, converted to a dataframe and stored in  a CSV
#       file.
def get_baseline_results (data, mpt, output_folder, n_grams):
    print("--- " + str(mpt) + "% ---")
    data = data[data.words_matched_percentage >= mpt]
    results = []

    results.append(run.run_knn_classification(data, None, "baseline", n_grams))
    results.append(run.run_decision_tree_classification(data, "baseline", n_grams))
    results.append(run.run_linear_svm_classification(data, None, "baseline", n_grams))
    results.append(run.run_naive_bayes_classification(data, None, "baseline", n_grams))
    results.append(run.run_random_forest_classification(data, None, "baseline", n_grams))

    results_df = pd.DataFrame(results, columns=["algorithm", "hyperparameter", "precision", "recall", "f-score", "experiment_type"])
    helpers.dataframe_to_csv(results_df, output_folder + str(mpt) + "_mpt_results.csv")
    print(results_df)

# negation_handled()
# parameters:
#   folder : string - the output folder
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function loads the dataset (negation handled), and calls the
#       get_baseline_results() function with varying values for the match
#       percentage threshold.
def negation_handled (folder, n_grams):
    data = helpers.load_dataset(ds.dataset + ds.negate_dataset)
    for mpt in range(0,100,10):
        get_baseline_results(data, mpt, folder, n_grams)

# negation_not_handled()
# parameters:
#   folder : string - the output folder
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function loads the dataset (negation not handled), and calls the
#       get_baseline_results() function with varying values for the match
#       percentage threshold.
def negation_not_handled (folder, n_grams):
    data = helpers.load_dataset(ds.dataset + ds.not_negate_dataset)
    for mpt in range(0,100,10):
        get_baseline_results(data, mpt, folder, n_grams)
