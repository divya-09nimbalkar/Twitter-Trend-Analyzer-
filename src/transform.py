import pandas as pd
import re


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"#", "", text)  # remove hashtags symbol
    return text.strip()


def transform_tweets(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Input data must include a text column.")
    transformed = df.copy()
    transformed["clean_text"] = transformed["text"].apply(clean_text)
    return transformed
