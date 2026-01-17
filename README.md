# Yt-Analysis

YouTube creator discovery pipeline for partner sourcing across 8 categories.
It replaces manual scouting with a repeatable, data-backed shortlist, so you
can quickly find relevant creators and compare performance without days of
searching.

## What it does
- Searches YouTube by category keywords (OR query)
- Collects unique channel IDs
- Fetches channel name + subscriber count + uploads playlist
- Estimates avg views from recent uploads
- Applies title-term relevance checks
- Ranks by relevance and exports top 50 per category

## Requirements
- Python 3.9+
- Google YouTube Data API v3 key

## Setup
1) (Optional) Create a virtualenv
```
python -m venv .venv
source .venv/bin/activate
```
2) Install dependencies
```
pip install google-api-python-client python-dotenv
```
3) Add your API key to `.env`
```
YOUTUBE_API_KEY=your_actual_key_here
```

## Run
```
python creator-discovery.py
```

## Output
One CSV per category (slugified name), each with:
- category
- channel_name
- channel_url
- subscribers
- avg_views_last_n

## Notes
- YouTube API quota is per project; new keys in the same project share quota.
- Tune `pages_per_keyword`, `include_channel_search`, and `n_recent_videos` in `creator-discovery.py` to trade recall vs. quota.
