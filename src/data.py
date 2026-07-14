import pandas as pd

from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE


def load_data(filepath: str) -> pd.DataFrame:

    df = pd.read_csv(filepath)
    return df


def train_test_split_data(df: pd.DataFrame, 
                            text_col: str = "Clean_Text",
                            label_col: str = "Label"):

    X = df[text_col].fillna("").astype(str)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        shuffle=True, stratify=y
    )

    return X_train, X_test, y_train, y_test