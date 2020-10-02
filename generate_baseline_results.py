import helpers
import dataset as ds
import pandas as pd
import run

# get_baseline_results()
# parameters:
#   data : DataFrame - the data containing the data for processing
#   mpt : int - the match percentage threshold to be retained of the dataset
#   output_folder : string - the folder path for storing CSV files
# returns:
#   None
# description:
#   This function creates baseline results for each of the five machine learning
#       algorithms by removing part of the dataset with a
#       words_matched_percentage value lower than the mpt variable. The results
#       are then stored in a list, converted to a dataframe and stored in  a CSV
#       file.
def get_baseline_results (data, mpt, output_folder):
    print("--- " + str(mpt) + "% ---")
    data = data[data.words_matched_percentage >= mpt]
    results = []

    results.append(run.run_knn_classification(data, None))
    results.append(run.run_decision_tree_classification(data))
    results.append(run.run_linear_svm_classification(data, None))
    results.append(run.run_naive_bayes_classification(data, None))
    results.append(run.run_random_forest_classification(data, None))

    results_df = pd.DataFrame(results, columns=["algorithm", "hyperparameter", "precision", "recall", "f-score"])
    helpers.dataframe_to_csv(results_df, output_folder + str(mpt) + "_mpt_results.csv")
    print(results_df)

# negation_handled()
# parameters:
#   None
# returns:
#   None
# description:
#   This function loads the dataset (negation handled), and calls the
#       get_baseline_results() function with varying values for the match
#       percentage threshold.
def negation_handled ():
    data = helpers.load_dataset(ds.dataset + ds.negate_dataset)
    get_baseline_results(data, 0, ds.negate_output)
    get_baseline_results(data, 10, ds.negate_output)
    get_baseline_results(data, 20, ds.negate_output)
    get_baseline_results(data, 30, ds.negate_output)
    get_baseline_results(data, 40, ds.negate_output)
    get_baseline_results(data, 50, ds.negate_output)
    get_baseline_results(data, 60, ds.negate_output)
    get_baseline_results(data, 70, ds.negate_output)
    get_baseline_results(data, 80, ds.negate_output)
    get_baseline_results(data, 90, ds.negate_output)

# negation_not_handled()
# parameters:
#   None
# returns:
#   None
# description:
#   This function loads the dataset (negation not handled), and calls the
#       get_baseline_results() function with varying values for the match
#       percentage threshold.
def negation_not_handled ():
    data = helpers.load_dataset(ds.dataset + ds.not_negate_dataset)
    get_baseline_results(data, 0, ds.not_negate_output)
    get_baseline_results(data, 10, ds.not_negate_output)
    get_baseline_results(data, 20, ds.not_negate_output)
    get_baseline_results(data, 30, ds.not_negate_output)
    get_baseline_results(data, 40, ds.not_negate_output)
    get_baseline_results(data, 50, ds.not_negate_output)
    get_baseline_results(data, 60, ds.not_negate_output)
    get_baseline_results(data, 70, ds.not_negate_output)
    get_baseline_results(data, 80, ds.not_negate_output)
    get_baseline_results(data, 90, ds.not_negate_output)

# execute ML for negation handled dataset
negation_handled()
# execute ML for negation not  handled dataset 
negation_not_handled()
