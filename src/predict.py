import joblib


def predict_sentiment(texts, model_path="models/sentiment_model.pkl"):
    model, vectorizer = joblib.load(model_path)
    X = vectorizer.transform(texts)
    return model.predict(X)
