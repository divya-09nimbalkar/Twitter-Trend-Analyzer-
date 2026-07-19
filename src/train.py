from __future__ import annotations

import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from src.transform import transform_tweets


def train_sentiment_model(
    df: pd.DataFrame | None = None,
    data_path: str = "data/raw/tweets.csv",
    model_path: str = "models/sentiment_model.pkl",
) -> str:
    if df is None:
        df = pd.read_csv(data_path)

    if "sentiment" not in df.columns:
        raise ValueError("Training data must include a sentiment column.")

    if "clean_text" not in df.columns:
        df = transform_tweets(df)

    X_texts = df["clean_text"]
    y = df["sentiment"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_texts)

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, vectorizer), output)
    return str(output)


if __name__ == "__main__":
    saved_path = train_sentiment_model()
    print(f"Sentiment model trained and saved at {saved_path}")
