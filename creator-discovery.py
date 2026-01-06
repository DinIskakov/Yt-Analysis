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


# ----------------------------
# Step 1: Search videos -> collect channel IDs
# ----------------------------

def discover_channel_ids(
    youtube,
    keywords: List[str],
    pages_per_keyword: int = 2,
    results_per_page: int = 50,
    order: str = "relevance",
) -> Set[str]:
    """
    For each keyword, search for videos, then grab channelId from each result.
    We dedupe early to avoid repeated work.
    """
    channel_ids: Set[str] = set()

    for kw in keywords:
        page_token = None

        for _ in range(pages_per_keyword):
            try:
                resp = youtube.search().list(
                    part="snippet",
                    q=kw,
                    type="video",
                    order=order,
                    maxResults=results_per_page,
                    pageToken=page_token,
                ).execute()
            except HttpError as e:
                print(f"[WARN] search failed for keyword={kw!r}: {e}")
                break

            for item in resp.get("items", []):
                cid = item.get("snippet", {}).get("channelId")
                if cid:
                    channel_ids.add(cid)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

            time.sleep(0.1)

    return channel_ids


# ----------------------------
# Step 2: Get channel name + subscribers (batch by 50)
# ----------------------------

def fetch_channel_metadata(
    youtube,
    channel_ids: List[str],
) -> Dict[str, Dict]:
    """
    Returns {channel_id: {"name": ..., "subscribers": ...}}
    subscriberCount can be hidden -> None.
    """
    out: Dict[str, Dict] = {}

    for batch in chunks(channel_ids, 50):
        try:
            resp = youtube.channels().list(
                part="snippet,statistics",
                id=",".join(batch),
                maxResults=50,
            ).execute()
        except HttpError as e:
            print(f"[WARN] channels.list failed for batch size={len(batch)}: {e}")
            continue

        for item in resp.get("items", []):
            cid = item["id"]
            name = item.get("snippet", {}).get("title", "")
            subs = safe_int(item.get("statistics", {}).get("subscriberCount"))
            out[cid] = {"name": name, "subscribers": subs}

        time.sleep(0.1)

    return out


