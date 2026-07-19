from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

TEXT_COLUMNS = ("text", "full_text", "tweet_text", "content")
DATE_COLUMNS = ("date", "created_at", "createdAt", "timestamp")
USER_COLUMNS = ("user", "username", "author_username", "screen_name")
SENTIMENT_COLUMNS = ("sentiment", "label", "predicted_sentiment")
DEFAULT_SENTIMENT = 2


def load_xquik_export(path: str | Path) -> pd.DataFrame:
    """Load a CSV, JSON, or JSONL Xquik export into the project tweet schema."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        frame = pd.DataFrame(_read_json_records(source))
    return normalize_tweet_frame(frame)


def normalize_tweet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return date, user, text, and sentiment columns for model training."""
    if frame.empty:
        raise ValueError("Tweet export is empty.")

    normalized = pd.DataFrame()
    normalized["text"] = _first_text_column(frame).astype(str).str.strip()
    normalized["date"] = _optional_column(frame, DATE_COLUMNS, "")
    normalized["user"] = _optional_column(frame, USER_COLUMNS, "unknown")
    normalized["sentiment"] = _normalize_sentiment(
        _optional_column(frame, SENTIMENT_COLUMNS, DEFAULT_SENTIMENT)
    )
    normalized = normalized[normalized["text"] != ""]
    if normalized.empty:
        raise ValueError("Tweet export does not contain usable tweet text.")
    return normalized.drop_duplicates(subset=["text"]).reset_index(drop=True)


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if path.suffix.lower() == ".jsonl":
        return [
            _flatten_record(json.loads(line))
            for line in raw.splitlines()
            if line.strip()
        ]

    payload = json.loads(raw)
    if isinstance(payload, dict):
        for key in ("tweets", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [_flatten_record(item) for item in value]
        return [_flatten_record(payload)]
    if isinstance(payload, list):
        return [_flatten_record(item) for item in payload]
    raise ValueError("JSON export must be an object, an array, or JSONL records.")


def _flatten_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    flattened = dict(record)
    author = record.get("author")
    if isinstance(author, dict):
        flattened.setdefault(
            "author_username", author.get("username") or author.get("screen_name")
        )
    return flattened


def _first_text_column(frame: pd.DataFrame) -> pd.Series:
    for column in TEXT_COLUMNS:
        if column in frame.columns:
            return frame[column]
    raise ValueError(
        f"Tweet export must include one of these text columns: {', '.join(TEXT_COLUMNS)}."
    )


def _optional_column(
    frame: pd.DataFrame, columns: tuple[str, ...], default: Any
) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column].fillna(default)
    return pd.Series([default] * len(frame), index=frame.index)


def _normalize_sentiment(values: pd.Series) -> pd.Series:
    labels = {
        "negative": 0,
        "neg": 0,
        "0": 0,
        "positive": 1,
        "pos": 1,
        "1": 1,
        "neutral": 2,
        "neu": 2,
        "2": 2,
    }
    return values.map(
        lambda value: labels.get(str(value).strip().lower(), DEFAULT_SENTIMENT)
    )
