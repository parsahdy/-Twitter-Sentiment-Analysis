import pandas as pd

from sklearn.metrics import (accuracy_score, recall_score,
                             f1_score, precision_score)



def evaluate_metrics(y_test, y_pred) -> str:
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    error_rate = 1 - accuracy
    f1 = f1_score(y_test,y_pred, average="macro")

    return {
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "Error Rate": round(error_rate, 2),
        "F1 Score": round(f1, 2)
    }
    