import json

import pandas as pd
import pytest

from src.xquik_import import load_xquik_export, normalize_tweet_frame


def test_normalize_tweet_frame_maps_xquik_columns():
    frame = pd.DataFrame(
        {
            "createdAt": ["2026-07-08T10:00:00Z"],
            "author_username": ["xquik"],
            "text": ["Xquik import works"],
            "predicted_sentiment": ["positive"],
        }
    )

    normalized = normalize_tweet_frame(frame)

    assert list(normalized.columns) == ["text", "date", "user", "sentiment"]
    assert normalized.loc[0, "date"] == "2026-07-08T10:00:00Z"
    assert normalized.loc[0, "user"] == "xquik"
    assert normalized.loc[0, "sentiment"] == 1


def test_load_xquik_json_export_flattens_author(tmp_path):
    export_path = tmp_path / "tweets.json"
    export_path.write_text(
        json.dumps(
            {
                "tweets": [
                    {
                        "createdAt": "2026-07-08T10:00:00Z",
                        "author": {"username": "brandops"},
                        "text": "Useful launch signal",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    normalized = load_xquik_export(export_path)

    assert normalized.loc[0, "user"] == "brandops"
    assert normalized.loc[0, "sentiment"] == 2


def test_normalize_tweet_frame_requires_text():
    with pytest.raises(ValueError, match="text columns"):
        normalize_tweet_frame(pd.DataFrame({"body": ["missing"]}))
