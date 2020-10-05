from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

# linear_svm_classifier()
# parameters:
#   c_value : integer/float - hyperparameter
#   training_instances_bow : matrix - a representation of the frequency of words
#       in each training instance in reference to all words in all training
#       instances
#   training_sentiment_scores : list - list of sentiment labels for the training
#       data
#   test_instances_bow : matrix - a representation of the frequency of words in
#       each test instance in reference to all words in all test instances
#   test_sentiment_scores : list - list of sentiment labels for the test data
# returns:
#   predicted_test_sentiment_scores : list - list of the predictions made by the
#       classifier
#   precision_recall_fscore_support() : tuple - the precision, recall, f_score
#       and support values
# description:
#   This function implements the Linear SVM classification algorithm, and
#       trains it on the training data (training_instances_bow). Once the
#       algorithm has been trained, it then makes predictions using the test
#       data (test_instances_bow). The precision, recall and f_score are
#       generated using the test predictions and the actual labels. These values
#       are then returned.
def linear_svm_classifier (c_value, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores):
    if c_value:
        classifier = LinearSVC(C=c_value, max_iter=10000)
    else:
        classifier = LinearSVC(max_iter=10000)
    classifier.fit(X=training_instances_bow, y=training_sentiment_scores)
    predicted_test_sentiment_scores = classifier.predict(test_instances_bow)
    return predicted_test_sentiment_scores, precision_recall_fscore_support(test_sentiment_scores, predicted_test_sentiment_scores, average='weighted')

# run()
# parameters:
#   modifier : integer/float - hyperparameter
#   training_instances_bow : matrix - a representation of the frequency of words
#       in each training instance in reference to all words in all training
#       instances
#   training_sentiment_scores : list - list of sentiment labels for the training
#       data
#   test_instances_bow : matrix - a representation of the frequency of words in
#       each test instance in reference to all words in all test instances
#   test_sentiment_scores : list - list of sentiment labels for the test data
# returns:
#   precision : float - a metric representing how precise the algorithm is
#       (true positives / true positives + false positives)
#   recall : float - a metric representing how recalling the algorithm is
#       (true positives / true positives + false negatives)
#   f_score : float - a combination of precision and recall
#       ((precision * recall) / (precision + recall)) * 2
# description:
#   This function takes the parameters required for the algorithm to run, calls
#       the classification function (linear_svm_classifier()) and extracts the
#       metrics from the returned data which are then returned to the
#       processor.data_split_bow_run() function.
def run (modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores):
    predictions, metrics = linear_svm_classifier(float(modifier), training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores)
    precision = round(metrics[0], 4)
    recall = round(metrics[1], 4)
    f_score = round(metrics[2], 4)
    return precision, recall, f_score
