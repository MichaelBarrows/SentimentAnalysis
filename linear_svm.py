from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

def linear_svm_classifier (c_value, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores):
    classifier = LinearSVC(C=c_value)
    classifier.fit(X=training_instances_bow, y=training_sentiment_scores)
    predicted_test_sentiment_scores = classifier.predict(test_instances_bow)
    return predicted_test_sentiment_scores, precision_recall_fscore_support(test_sentiment_scores, predicted_test_sentiment_scores, average='weighted')

def run (modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores):
    predictions, metrics = linear_svm_classifier(modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores)
    precision = round(metrics[0], 4)
    recall = round(metrics[1], 4)
    f_score = round(metrics[2], 4)
    return precision, recall, f_score
