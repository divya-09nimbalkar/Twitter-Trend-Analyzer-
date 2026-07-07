import pandas as pd
import re

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"#", "", text)        # remove hashtags symbol
    return text.strip()

def transform_tweets(df: pd.DataFrame) -> pd.DataFrame:
    df["clean_text"] = df["text"].apply(clean_text)
    return df
