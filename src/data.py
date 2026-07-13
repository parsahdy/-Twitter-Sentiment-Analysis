import pandas as pd

from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:

    df = pd.read_csv(filepath)
    return df


def train_test_split_data(df: pd.DataFrame, 
                            text_col: str = "Clean_Text",
                            label_col: str = "Label"):

    X = df[text_col].fillna("").astype(str)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        shuffle=True, stratify=y
    )

    return X_train, X_test, y_train, y_test