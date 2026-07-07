import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def train_sentiment_model():
    # Load your sample tweets dataset
    df = pd.read_csv("data/raw/tweets.csv")

    # Preprocess: just use text + sentiment
    X_texts = df["text"]
    y = df["sentiment"]

    # Convert text to bag-of-words
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_texts)

    # Train simple logistic regression
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Save both model + vectorizer
    joblib.dump((model, vectorizer), "models/sentiment_model.pkl")
    print("✅ Sentiment model trained and saved at models/sentiment_model.pkl")

if __name__ == "__main__":
    train_sentiment_model()
