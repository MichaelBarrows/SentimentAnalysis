import helpers
import dataset as ds
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

# import_results()
# parameters:
#   folder : string - the folder for results to be imported from
# returns:
#   None
# description:
#   This function gets the output results files, loops over them, sorts them,
#       adds the first sorted row (highest f-score) to a new dataframe and
#       stores that dataframe.
def import_results (folder):
    new_results = []
    files = get_results_filenames(folder)
    for file in files:
        print("---" + file + "---")
        mpt = file.split("_")[0]
        if mpt == "best":
            continue
        mpt = int(mpt)
        results_df = helpers.load_dataset(folder + file)
        results_df = results_df.sort_values(['f-score'],ascending=False)
        results_df = results_df.reset_index(drop=True)
        for index, row in results_df.iterrows():
            new_results.append([mpt, row.algorithm, row.hyperparameter, row.precision, row.recall, row['f-score'], row.experiment_type])
            break
    new_results_df = pd.DataFrame(new_results, columns=["mpt", "algorithm", "hyperparameter", "precision", "recall", "f-score", "experiment_type"])
    helpers.dataframe_to_csv(new_results_df, folder + "best_result_per_mpt.csv")

# import import_best_results_and_sort()
# parameters:
#   folder : string - the folder path that the files are stored in
# returns:
#   None
# description:
#   This function imports the best results file output in the previous function
#       as a dataframe, sorts it by highest f-score and stores the sorted
#       dataframe
def import_best_results_and_sort (folder):
    best_results_df = helpers.load_dataset(folder + "best_result_per_mpt.csv")
    best_results_df = best_results_df.sort_values(['f-score'],ascending=False)
    best_results_df = best_results_df.reset_index(drop=True)
    helpers.dataframe_to_csv(best_results_df, folder + "best_result_per_mpt_sorted.csv")
