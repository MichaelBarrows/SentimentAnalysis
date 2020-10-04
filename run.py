import processor

n_folds = 10

# run_knn_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   hyperparameter : int - the hyperparameter used to modify the algorithm
#   experiment_type : string - the string detailing the experiment type
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the K-Nearest Neighbours classification algorithm. The results
#       of the execution are then returned returned.
def run_knn_classification (df, hyperparameter, experiment_type):
    global n_folds
    data = processor.data_split_bow_run("knn", hyperparameter, n_folds, df)
    if not hyperparameter:
        hyperparameter = "default"
    return ["KNN", hyperparameter, data[0], data[1], data[2], experiment_type]

# run_decision_tree_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   experiment_type : string - the string detailing the experiment type
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Decision Tree classification algorithm. The results of the
#       execution are then returned returned.
def run_decision_tree_classification (df, experiment_type):
    global n_folds
    data = processor.data_split_bow_run("decision_tree", None, n_folds, df)
    return ['Decision Tree', "default", data[0], data[1], data[2], experiment_type]

# run_random_forest_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   hyperparameters : list - the hyperparameters used to modify the algorithm
#   experiment_type : string - the string detailing the experiment type
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Random Forest classification algorithm. The results of the
#       execution are then returned returned.
def run_random_forest_classification (df, hyperparameters, experiment_type):
    global n_folds
    data = processor.data_split_bow_run("random_forest", hyperparameters, n_folds, df)
    if not hyperparameters:
        hyperparameters = "default"
    return ["Random Forest", hyperparameters, data[0], data[1], data[2], experiment_type]


# run_naive_bayes_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   hyperparameter : int - the hyperparameter used to modify the algorithm
#   experiment_type : string - the string detailing the experiment type
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Naive Bayes classification algorithm. The results of the
#       execution are then returned returned.
def run_naive_bayes_classification (df, hyperparameter, experiment_type):
    global n_folds
    data = processor.data_split_bow_run("naive_bayes", hyperparameter, n_folds, df)
    if not hyperparameter:
        hyperparameter = "default"
    return ["Naive Bayes", hyperparameter, data[0], data[1], data[2], experiment_type]

# run_linear_svm_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   hyperparameter : int - the hyperparameter used to modify the algorithm
#   experiment_type : string - the string detailing the experiment type
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Linear Support Vector Machine (SVM) classification
#       algorithm. The results of the execution are then returned returned.
def run_linear_svm_classification (df, hyperparameter, experiment_type):
    global n_folds
    data = processor.data_split_bow_run("linear_svm", hyperparameter, n_folds, df)
    if not hyperparameter:
        hyperparameter = "default"
    return ["Linear SVM", hyperparameter, data[0], data[1], data[2], experiment_type]
