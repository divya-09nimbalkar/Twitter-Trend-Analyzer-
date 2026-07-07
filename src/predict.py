import joblib

def predict_sentiment(texts):
    model, vectorizer = joblib.load("models/sentiment_model.pkl")
    X = vectorizer.transform(texts)
    return model.predict(X)
