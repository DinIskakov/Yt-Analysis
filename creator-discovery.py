from __future__ import annotations

import os
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Iterable, Optional

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ----------------------------
# Data model (simple + readable)
# ----------------------------

@dataclass
class ChannelInfo:
    channel_id: str
    name: str
    url: str
    subscribers: Optional[int]
    avg_views_last_n: Optional[int]


# ----------------------------
# Small helpers
# ----------------------------

def chunks(items: List[str], size: int) -> Iterable[List[str]]:
    """Split a list into chunks of length <= size (YouTube API often limits IDs to 50)."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def safe_int(value) -> Optional[int]:
    """Convert API numeric strings to int, return None if missing/hidden."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ----------------------------
# YouTube client (API key auth)
# ----------------------------

def youtube_client_from_env():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing YOUTUBE_API_KEY. Put it in a .env file or export it in your shell."
        )

    # build() is the official Google client method
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


