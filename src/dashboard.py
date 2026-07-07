import streamlit as st
import pandas as pd
from src.predict import predict_sentiment

st.title("Twitter Trend Analyzer")

uploaded_file = st.file_uploader("Upload CSV of tweets", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Tweets", df.head())
    preds = predict_sentiment(df["text"])
    df["predicted_sentiment"] = preds
    st.write("Predictions", df)
