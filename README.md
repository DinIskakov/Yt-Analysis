# Yt-Analysis

Discover YouTube creators by keyword, enrich them with channel stats, and export
results to CSV.

## What it does
- Searches YouTube videos for a list of keywords
- Collects unique channel IDs
- Fetches channel name + subscriber count
- Estimates average views from recent videos
- Writes a CSV you can review or filter

## Requirements
- Python 3.9+
- Google YouTube Data API v3 key

## Setup
1) Create a virtualenv (optional but recommended)
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
The script writes a CSV named `design_architecture_channels.csv` with:
- category
- channel_name
- channel_url
- subscribers
- avg_views_last_n

## Notes
- YouTube API quotas apply; increase pages gradually.
- Subscriber counts can be hidden for some channels.
