import pandas as pd
import contractions
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

URL_PATTERN = r"https?://\S+"
MENTION_PATTERN = r"@\w+"
HASHTAG_PATTERN = r"#"
PUNCTUATION_PATTERN = r"[!?_]"
APOSTROPHE_PATTERN = r"'s\b"


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "Clean_Text",
                         tokens: str = "Tokens") -> pd.DataFrame:
 
    df = df.copy()

    # lowercase
    df[text_col] = df[text_col].fillna("").astype(str).str.lower()

    # remove urls
    df[text_col] = df[text_col].fillna("").astype(str).str.replace(URL_PATTERN, "", regex=True)
    
    # remove mentions 
    df[text_col] = df[text_col].fillna("").astype(str).str.replace(MENTION_PATTERN, "", regex=True)
    
    # remove hashtgs
    df[text_col] = df[text_col].fillna("").astype(str).str.replace(HASHTAG_PATTERN, "", regex=True)

    # remove punctuations
    df[text_col] = df[text_col].fillna("").astype(str).str.replace(PUNCTUATION_PATTERN, "", regex=True)

    # remove apostorophe
    df[text_col] = df[text_col].fillna("").astype(str).str.replace(APOSTROPHE_PATTERN, "", regex=True)

    # remove extra spaces
    df[text_col] = (df[text_col].fillna("").astype(str).apply(
                    lambda x: " ".join(x.split())
    ))

    # contractions
    df[text_col] = (df[text_col].fillna("").astype(str).apply(
                    lambda x: contractions.fix(x)
    ))

    # tokenize
    df[tokens] = df[text_col].fillna("").astype(str).apply(word_tokenize)

    # remove stopwords
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")

    df[tokens] = (df[tokens].apply(
                  lambda x: [word for word in x if word.lower() not in stop_words]
    ))

    df[text_col] = df[tokens].apply(lambda x: " ".join(x))

    return df