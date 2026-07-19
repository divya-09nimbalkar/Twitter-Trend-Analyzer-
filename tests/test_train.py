import pandas as pd
from src.predict import predict_sentiment
from src.train import train_sentiment_model


def test_train_sentiment_model(tmp_path):
    df = pd.DataFrame(
        {"clean_text": ["good product", "bad service"], "sentiment": [1, 0]}
    )
    model_path = tmp_path / "sentiment_model.pkl"
    saved_path = train_sentiment_model(df, model_path=str(model_path))
    assert saved_path == str(model_path)
    assert model_path.exists()

    predictions = predict_sentiment(
        ["good product", "bad service"], model_path=str(model_path)
    )
    assert predictions.tolist() == [1, 0]
