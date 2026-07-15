import mlflow
import mlflow.sklearn

from sklearn.svm import LinearSVC
from sklearn.metrics import (f1_score, accuracy_score,
                            precision_score, recall_score)
from sklearn.feature_extraction.text import TfidfVectorizer

from data import load_data, train_test_split_data
from config import MAX_FEATURES, FILE_PATH



df = load_data(FILE_PATH)
X_train, X_test, y_train, y_test = train_test_split_data(df, text_col="Clean_Text", label_col="Label")


tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=MAX_FEATURES
)

X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)


mlflow.set_experiment("sentiment140-baselines")


for C in [0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"LinearSVC_C={C}"):

        model = LinearSVC(C=C)

        model.fit(X_train_tf, y_train)

        preds = model.predict(X_test_tf)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }

        params = {
            "model": "LinearSVC",
            "vectorizer": "tfidf",
            "max_features": MAX_FEATURES,
            "C": C
        }

        for k, v in params.items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, "model")

        print(f"Baseline {C}, metrics: {metrics}")