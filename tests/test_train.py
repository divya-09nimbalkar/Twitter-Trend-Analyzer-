import pandas as pd
from src.train import train_sentiment_model
import os

def test_train_sentiment_model(tmp_path):
    df = pd.DataFrame({
        "clean_text": ["good product", "bad service"],
        "sentiment": [1, 0]
    })
    train_sentiment_model(df)
    assert os.path.exists("models/sentiment_model.pkl")
