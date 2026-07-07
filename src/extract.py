import tweepy
import pandas as pd

def fetch_tweets(query: str, api, count: int = 100) -> pd.DataFrame:
    """Fetch tweets using Tweepy API."""
    tweets = api.search_tweets(q=query, count=count, lang="en", tweet_mode="extended")
    data = [{"date": t.created_at, "user": t.user.screen_name, "text": t.full_text} for t in tweets]
    return pd.DataFrame(data)
