import helpers
import dataset as ds
import run
import pandas as pd

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

# process_experiments()
# parameters:
#   data : DataFrame - dataframe containing the twitter data
#   mpt : int - the match percentage threshold for the data`
#   experiments : DataFrame - dataframe containing details of experiments to be
#       conducted
#   results_df : DataFrame - dataframe containing existing results
# returns:
#   results_df : DataFrame - dataframe containing existing and new results
# description:
#   This function retains the data above the mpt threshold and iterates over the
#       experiments dataframe. For each row in the experiments df, it executes
#       the given experiment (and modifies the hyperparameter for random forest)
#       and stores the results. The results are then added to the results df,
#       which is then returned.
def process_experiments (data, mpt, experiments, results_df):
    data = data[data.words_matched_percentage >= mpt]
    results = []
    counter = 1
    for index, row in experiments.iterrows():
        print(str(mpt) + "% - " + str(counter) + " / " + str(len(experiments)))
        counter += 1
        if row.algorithm == "KNN":
            results.append(run.run_knn_classification(data, row.hyperparameter, "experiment"))
        elif row.algorithm == "Linear SVM":
            results.append(run.run_linear_svm_classification(data, row.hyperparameter, "experiment"))
        elif row.algorithm == "Naive Bayes":
            results.append(run.run_naive_bayes_classification(data, row.hyperparameter, "experiment"))
        elif row.algorithm == "Random Forest":
            hyperparameter = row.hyperparameter.split(', ')
            results.append(run.run_random_forest_classification(data, [int(hyperparameter[0]), int(hyperparameter[1])], "experiment"))
    results = pd.DataFrame(results, columns=["algorithm", "hyperparameter", "precision", "recall", "f-score", "experiment_type"])
    results_df = results_df.append(results)
    results_df = results_df.reset_index(drop=True)
    print(results_df)
    return results_df

# process_negation_handled_experiments()
# parameters:
#   None
# returns:
#   None
# description:
#   This function gets the filenames for the results files, imports the twitter
#       data and the experiments data. The results files are looped over, with
#       the results data for the given MPT imported and experiments within that
#       MPT are retained. The process_experiments() function is then called to
#       execute the experiments, and the results data is stored.
def process_negation_handled_experiments ():
    results_files = get_results_filenames(ds.negate_output)
    data = helpers.load_dataset(ds.dataset + ds.negate_dataset)
    experiments_df = helpers.load_dataset("/home/michael/MRes/actual_project/sentiment_analysis/next_negation_handled_experiments.csv")
    for results_filename in results_files:
        mpt = int(results_filename.split("_")[0])
        experiments = experiments_df[experiments_df.mpt == mpt]
        results_df = helpers.load_dataset(ds.negate_output + results_filename)
        results_df = process_experiments(data, mpt, experiments, results_df)
        helpers.dataframe_to_csv(results_df, ds.negate_output + results_filename)
    return

# process_negation_not_handled_experiments()
# parameters:
#   None
# returns:
#   None
# description:
#   This function gets the filenames for the results files, imports the twitter
#       data and the experiments data. The results files are looped over, with
#       the results data for the given MPT imported and experiments within that
#       MPT are retained. The process_experiments() function is then called to
#       execute the experiments, and the results data is stored.
def process_negation_not_handled_experiments ():
    results_files = get_results_filenames(ds.not_negate_output)
    data = helpers.load_dataset(ds.dataset + ds.not_negate_dataset)
    experiments_df = helpers.load_dataset("/home/michael/MRes/actual_project/sentiment_analysis/next_negation_not_handled_experiments.csv")
    for results_filename in results_files:
        mpt = int(results_filename.split("_")[0])
        experiments = experiments_df[experiments_df.mpt == mpt]
        results_df = helpers.load_dataset(ds.not_negate_output + results_filename)
        results_df = process_experiments(data, mpt, experiments, results_df)
        helpers.dataframe_to_csv(results_df, ds.not_negate_output + results_filename)
    return

# execute negation handled experiments
process_negation_handled_experiments()
# execute negation not handled experiments
process_negation_not_handled_experiments()
