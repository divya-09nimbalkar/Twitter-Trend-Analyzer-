"""Command-line entry point for the Twitter Trend Analyzer pipeline."""

import argparse
from pathlib import Path

from src.predict import predict_sentiment
from src.train import train_sentiment_model
from src.transform import transform_tweets
from src.xquik_import import load_xquik_export


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the tweet sentiment training pipeline."
    )
    parser.add_argument(
        "--input", default="data/raw/tweets.csv", help="Raw tweet CSV path."
    )
    parser.add_argument(
        "--xquik-export", help="Optional Xquik CSV, JSON, or JSONL tweet export."
    )
    parser.add_argument("--processed-output", default="data/processed/tweets_clean.csv")
    parser.add_argument(
        "--model-path",
        default="models/sentiment_model.pkl",
        help="Model to use for Xquik results or write after training.",
    )
    args = parser.parse_args()

    raw = load_xquik_export(args.xquik_export or args.input)
    processed = transform_tweets(raw)
    if args.xquik_export:
        processed["predicted_sentiment"] = predict_sentiment(
            processed["clean_text"], model_path=args.model_path
        )

    processed_path = Path(args.processed_output)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)
    print(f"Processed tweets saved to {processed_path}")

    if args.xquik_export:
        print(f"Sentiment predictions created with {args.model_path}")
    else:
        model_path = train_sentiment_model(processed, model_path=args.model_path)
        print(f"Sentiment model saved to {model_path}")


if __name__ == "__main__":
    main()
