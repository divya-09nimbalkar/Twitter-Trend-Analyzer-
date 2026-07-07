import pandas as pd
from src.transform import transform_tweets

def test_transform_tweets():
    df = pd.DataFrame({"text": ["Hello @user check http://example.com #trend"]})
    df_clean = transform_tweets(df)
    assert "clean_text" in df_clean.columns
    assert "http" not in df_clean["clean_text"].iloc[0]
