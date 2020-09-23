from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import knn
import decision_tree
import random_forest
import naive_bayes
import linear_svm


def data_split_bow_run (algorithm, modifier, n_folds, df):
    kf = KFold(n_splits=n_folds)
    datastore = []
    counter = 1
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
        datastore.append([training_ids, training_texts, training_sentiment_scores, test_ids, test_texts, test_sentiment_scores])

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
        print(counter)
        counter += 1
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

def average (list):
    total = 0
    for item in list:
        total += item
    return round(total / len(list), 4)
