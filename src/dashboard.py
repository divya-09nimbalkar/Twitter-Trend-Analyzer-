import streamlit as st
import pandas as pd
from src.predict import predict_sentiment
from src.xquik_import import normalize_tweet_frame

st.title("Twitter Trend Analyzer")

uploaded_file = st.file_uploader("Upload CSV of tweets", type="csv")
if uploaded_file:
    try:
        df = normalize_tweet_frame(pd.read_csv(uploaded_file))
        st.write("Raw Tweets", df.head())
        preds = predict_sentiment(df["text"])
        df["predicted_sentiment"] = preds
        st.write("Predictions", df)
    except ValueError as exc:
        st.error(str(exc))
