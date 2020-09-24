import helpers
import dataset as ds
import processor

df = helpers.load_dataset(ds.dataset + ds.sentiwordnet_unclassified_removed)
n_folds = 10

# run_knn_classification()
# parameters:
#   None
# returns:
#   storage : list - a list containing results returned from the classification
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the K-Nearest Neighbours classification algorithm. The results
#       of the execution are stored and the list of stored results is returned.
def run_knn_classification ():
    global df, n_folds
    storage = []
    k_neighbours = [1,3,5,7,9,11]
    for k in k_neighbours:
        storage.append([k, processor.data_split_bow_run("knn", k, n_folds, df)])
    return storage

# run_decision_tree_classification()
# parameters:
#   None
# returns:
#   storage : list - a list containing results returned from the classification
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Decision Tree classification algorithm. The results of the
#       execution are stored and the list of stored results is returned.
def run_decision_tree_classification ():
    global df, n_folds
    storage = []
    storage.append(['DT', processor.data_split_bow_run("decision_tree", None, n_folds, df)])
    return storage

# run_random_forest_classification()
# parameters:
#   None
# returns:
#   storage : list - a list containing results returned from the classification
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Random Forest classification algorithm. The results of the
#       execution are stored and the list of stored results is returned.
def run_random_forest_classification ():
    global df, n_folds
    storage = []
    trees_features = []
    for tree_feature in trees_features:
        storage.append([tree_feature, processor.data_split_bow_run("random_forest", tree_feature, n_folds, df)])
    return storage

# run_naive_bayes_classification()
# parameters:
#   None
# returns:
#   storage : list - a list containing results returned from the classification
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Naive Bayes classification algorithm. The results of the
#       execution are stored and the list of stored results is returned.
def run_naive_bayes_classification ():
    global df, n_folds
    storage = []
    alpha_values = []
    for alpha_value in alpha_values:
        storage.append([alpha_value, processor.data_split_bow_run("naive_bayes", alpha_value, n_folds, df)])
    return storage

# run_linear_svm_classification()
# parameters:
#   None
# returns:
#   storage : list - a list containing results returned from the classification
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Linear Support Vector Machine (SVM) classification
#       algorithm. The results of the execution are stored and the list of
#       stored results is returned.
def run_linear_svm_classification ():
    global df, n_folds
    storage = []
    c_values = [0.3]
    for c in c_values:
        storage.append([c, processor.data_split_bow_run("linear_svm", c, n_folds, df)])
    return storage


# Call the algorithms and store the results
knn_data = run_knn_classification()
decision_tree_data = run_decision_tree_classification()
random_forest_data = run_random_forest_classification()
naive_bayes_data = run_naive_bayes_classification()
linear_svm_data = run_linear_svm_classification()

print(knn_data)
print(decision_tree_data)
print(random_forest_data)
print(naive_bayes_data)
print(linear_svm_data)
