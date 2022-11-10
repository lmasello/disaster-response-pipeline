import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import (
    f1_score, make_scorer, precision_score, 
    recall_score, classification_report
)


def mean_score(scoring, y_true, y_pred, **kwargs):
    """Compute a scoring function iterating over the target categories"""
    acc = []
    for col in range(y_pred.shape[1]):
        acc.append(scoring(y_true.iloc[:, col], y_pred[:, col], **kwargs))
    return np.mean(acc)    


def mean_f1(y_true, y_pred):
    """Compute the mean f1 score over the target categories"""
    return mean_score(f1_score, y_true, y_pred, average='micro')


def mean_precision(y_true, y_pred):
    """Compute the mean precision score over the target categories"""
    return mean_score(precision_score, y_true, y_pred, average='micro')
    
    
def mean_recall(y_true, y_pred):
    """Compute the mean recall score over the target categories"""
    return mean_score(recall_score, y_true, y_pred, average='micro')        


def display_results(y_true, y_pred, category_names):
    """Display a sklearn's classification_report for each category"""
    for col, category in enumerate(category_names):
        print('Classification report for:', category)
        print(classification_report(y_true.iloc[:, col], y_pred[:, col]))

        
def overall_scores(y_true, y_pred):
    """Print performance metrics of the estimator over all categories"""
    print('F1 score:', mean_f1(y_true, y_pred))
    print('Precision score:', mean_precision(y_true, y_pred))
    print('Recall score:', mean_recall(y_true, y_pred))
    
    
mean_f1_score = make_scorer(mean_f1)
mean_precision_score = make_scorer(mean_precision)
mean_recall_score = make_scorer(mean_recall)