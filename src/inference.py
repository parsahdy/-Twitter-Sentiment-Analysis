import joblib
import pandas as pd
from preprocess import preprocess_dataframe



model = joblib.load("models/svm.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predicte_sentiment(df: pd.DataFrame, text_col: str = "Clean_Text") -> pd.DataFrame:
    
    df_clean = preprocess_dataframe(df, text_col=text_col)

    texts = df_clean[text_col].astype(str).tolist()

    vectore = vectorizer.transform(texts)

    prediction = model.predict(vectore)

    df_clean["prediction"] = prediction
    df_clean["label"] = df_clean["prediction"].map(
        {1: "Positive", 0: "Negative"}
    )


    return df_clean