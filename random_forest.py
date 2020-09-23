from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

def random_forest_classifier (trees, features, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores):
    classifier = RandomForestClassifier(n_estimators=trees,
                                    max_features=features,
                                    n_jobs=100)
    classifier.fit(training_instances_bow, training_sentiment_scores)
    predicted_test_sentiment_scores = classifier.predict(test_instances_bow)
    return predicted_test_sentiment_scores, precision_recall_fscore_support(test_sentiment_scores, predicted_test_sentiment_scores, average='weighted')

def run (modifier, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores):
    trees = modifier[0]
    features = modifier[1]
    predictions, metrics = random_forest_classifier(trees, features, training_instances_bow, training_sentiment_scores, test_instances_bow, test_sentiment_scores)
    precision = round(metrics[0], 4)
    recall = round(metrics[1], 4)
    f_score = round(metrics[2], 4)
    return precision, recall, f_score
