from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support

# naive_bayes_classifier()
# parameters:
#   alpha : integer/float - hyperparameter
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
#   This function implements the Naive Bayes classification algorithm, and
#       trains it on the training data (training_instances_bow). Once the
#       algorithm has been trained, it then makes predictions using the test
#       data (test_instances_bow). The precision, recall and f_score are
#       generated using the test predictions and the actual labels. These values
#       are then returned.
def naive_bayes_classifier (alpha, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores):
    if alpha:
        classifier = MultinomialNB(alpha=alpha)
    else:
        classifier = MultinomialNB()
    classifier.fit(training_instances_bow, training_sentiment_scores)
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
#       the classification function (naive_bayes_classifier()) and extracts the
#       metrics from the returned data which are then returned to the
#       processor.data_split_bow_run() function.
def run (modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores):
    predictions, metrics = naive_bayes_classifier(modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores)
    precision = round(metrics[0], 4)
    recall = round(metrics[1], 4)
    f_score = round(metrics[2], 4)
    return precision, recall, f_score
