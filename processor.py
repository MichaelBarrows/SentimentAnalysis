from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import knn
import decision_tree
import random_forest
import naive_bayes
import linear_svm

# data_split_bow_run()
# parameters:
#   algorithm : string - the name of the algorithm to be executed
#   modifier : integer/float - the modifier value (hyperparameter) for the
#       algorithm
#   n_folds : integer - the number of folds that the data will be split into
#   df : DataFrame - a dataframe containing the data to be split and used
# returns:
#   precision : float - a metric representing how precise the algorithm is
#       (true positives / true positives + false positives)
#   recall : float - a metric representing how recalling the algorithm is
#       (true positives / true positives + false negatives)
#   f_score : float - a combination of precision and recall
#       ((precision * recall) / (precision + recall)) * 2
# description:
#   This function implements the process of splitting data into folds for
#       training and testing, extracting the text and sentiment labels,
#       converting the text to a bag of words representation. This function then
#       calls the relevant algorithm to be used, averaging the precisions,
#       recalls and f_scores across all of the folds to return the final metrics.
def data_split_bow_run (algorithm, modifier, n_folds, df):
    kf = KFold(n_splits=n_folds)
    print(algorithm, modifier)
    for training_index, test_index in kf.split(df.index.tolist()):
        training_ids, training_texts, training_sentiment_scores  = [], [], []
        test_ids, test_texts, test_sentiment_scores = [], [], []
        for index, row in df.iterrows():
            if index in training_index:
                training_ids.append(index)
                training_texts.append(str(row.preprocessed_tweet_text))
                training_sentiment_scores.append(str(row.sentiment_class))
            elif index in test_index:
                test_ids.append(index)
                test_texts.append(str(row.preprocessed_tweet_text))
                test_sentiment_scores.append(str(row.sentiment_class))

        training_vectorizer = CountVectorizer()
        training_vectorizer.fit(training_texts)
        training_instances_bow = training_vectorizer.transform(training_texts)

        # convert test text reviews into bag-of-words (bow)
        test_vectorizer = CountVectorizer(vocabulary=training_vectorizer.get_feature_names())
        test_vectorizer.fit(test_texts)
        test_instances_bow = test_vectorizer.fit_transform(test_texts)

        # call algorithm
        precision = []
        recall = []
        f_score = []
        if algorithm == "knn":
            precision_data, recall_data, f_score_data = knn.run(modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores)
        elif algorithm == "decision_tree":
            precision_data, recall_data, f_score_data = decision_tree.run(training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores)
        elif algorithm == "random_forest":
            precision_data, recall_data, f_score_data = random_forest.run(modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores)
        elif algorithm == "naive_bayes":
            precision_data, recall_data, f_score_data = naive_bayes.run(modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores)
        elif algorithm == "linear_svm":
            precision_data, recall_data, f_score_data = linear_svm.run(modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores)
        else:
            return

        precision.append(precision_data)
        recall.append(recall_data)
        f_score.append(f_score_data)
    precision = average(precision)
    recall = average(recall)
    f_score = average(f_score)
    return precision, recall, f_score

# average():
# parameters:
#   list : list - list of values to be averaged
# returns:
#   integer - the rounded (to 4 decimal places) average of all values in the
#       list
# description:
#   This function totals the items in the list and divides them by the length
#       of the list to generate an average which is then rounded to 4 decimal
#       places and returned.
def average (list):
    total = 0
    for item in list:
        total += item
    return round(total / len(list), 4)
