from __future__ import annotations

import os
import csv
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Iterable, Optional, Tuple

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


@dataclass
class VideoStat:
    title: str
    view_count: Optional[int]


@dataclass
class ScoredChannel:
    info: ChannelInfo
    hits: int
    title_matches: int


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def slugify_category(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "category"


def term_in_text(text: str, term: str) -> bool:
    term = normalize_text(term)
    text = normalize_text(text)
    if not term:
        return False
    if " " in term:
        return term in text
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


def count_title_matches(video_stats: List[VideoStat], terms: List[str]) -> int:
    if not terms:
        return 0
    matches = 0
    for v in video_stats:
        if any(term_in_text(v.title, term) for term in terms):
            matches += 1
    return matches


def select_candidate_ids(
    hit_counts: Dict[str, int],
    min_hits: int,
    max_candidates: int,
) -> List[str]:
    candidates = [(cid, hits) for cid, hits in hit_counts.items() if hits >= min_hits]
    candidates.sort(key=lambda item: item[1], reverse=True)
    if max_candidates and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    return [cid for cid, _ in candidates]


def relevance_sort_key(scored: ScoredChannel) -> Tuple[int, int, int]:
    avg_views = scored.info.avg_views_last_n or -1
    return (scored.title_matches, scored.hits, avg_views)


def passes_quality_filters(
    scored: ScoredChannel,
    min_subscribers: int,
    min_avg_views: int,
    min_view_to_sub_ratio: float,
) -> bool:
    subs = scored.info.subscribers
    avg_views = scored.info.avg_views_last_n
    if min_subscribers and (subs is None or subs < min_subscribers):
        return False
    if min_avg_views and (avg_views is None or avg_views < min_avg_views):
        return False
    if min_view_to_sub_ratio:
        if not subs or avg_views is None:
            return False
        if (avg_views / subs) < min_view_to_sub_ratio:
            return False
    return True


# ----------------------------
# YouTube client (API key auth)
# ----------------------------

def youtube_client_from_env():
    # Ensure .env overrides any previously exported shell variable.
    load_dotenv(override=True)
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

def record_hit(
    channel_id: Optional[str],
    keyword: str,
    hit_counts: Dict[str, int],
    matched_keywords: Dict[str, Set[str]],
    seen_this_keyword: Set[str],
) -> None:
    if not channel_id or channel_id in seen_this_keyword:
        return
    hit_counts[channel_id] = hit_counts.get(channel_id, 0) + 1
    matched_keywords.setdefault(channel_id, set()).add(keyword)
    seen_this_keyword.add(channel_id)


def discover_channel_ids(
    youtube,
    keywords: List[str],
    pages_per_keyword: int = 2,
    results_per_page: int = 50,
    order: str = "relevance",
    include_channel_search: bool = True,
    channel_pages_per_keyword: int = 1,
    use_or_query: bool = False,
) -> Tuple[Dict[str, int], Dict[str, Set[str]]]:
    """
    For each keyword, search for videos, then grab channelId from each result.
    We count a max of 1 hit per keyword per channel to reduce noise.
    """
    hit_counts: Dict[str, int] = {}
    matched_keywords: Dict[str, Set[str]] = {}

    query_terms = keywords
    if use_or_query and keywords:
        quoted = [f"\"{kw}\"" if " " in kw else kw for kw in keywords]
        query_terms = [" OR ".join(quoted)]

    for kw in query_terms:
        seen_this_keyword: Set[str] = set()
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
                record_hit(cid, kw, hit_counts, matched_keywords, seen_this_keyword)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

            time.sleep(0.1)

        if include_channel_search and channel_pages_per_keyword > 0:
            page_token = None
            for _ in range(channel_pages_per_keyword):
                try:
                    resp = youtube.search().list(
                        part="snippet",
                        q=kw,
                        type="channel",
                        order=order,
                        maxResults=results_per_page,
                        pageToken=page_token,
                    ).execute()
                except HttpError as e:
                    print(f"[WARN] channel search failed for keyword={kw!r}: {e}")
                    break

                for item in resp.get("items", []):
                    cid = item.get("id", {}).get("channelId")
                    record_hit(cid, kw, hit_counts, matched_keywords, seen_this_keyword)

                page_token = resp.get("nextPageToken")
                if not page_token:
                    break

                time.sleep(0.1)

    return hit_counts, matched_keywords


# ----------------------------
# Step 2: Get channel name + subscribers (batch by 50)
# ----------------------------

def fetch_channel_metadata(
    youtube,
    channel_ids: List[str],
) -> Dict[str, Dict]:
    """
    Returns {channel_id: {"name": ..., "subscribers": ..., "uploads_playlist_id": ...}}
    subscriberCount can be hidden -> None.
    """
    out: Dict[str, Dict] = {}

    for batch in chunks(channel_ids, 50):
        try:
            resp = youtube.channels().list(
                part="snippet,statistics,contentDetails",
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
            uploads = (
                item.get("contentDetails", {})
                .get("relatedPlaylists", {})
                .get("uploads")
            )
            out[cid] = {
                "name": name,
                "subscribers": subs,
                "uploads_playlist_id": uploads,
            }

        time.sleep(0.1)

    return out


# ----------------------------
# Step 3: Estimate avg views from last N videos
# ----------------------------

def recent_video_stats_for_channel(
    youtube,
    uploads_playlist_id: Optional[str],
    n_recent: int = 10,
) -> List[VideoStat]:
    """Fetch titles + view counts for up to N most recent videos."""
    if not uploads_playlist_id:
        return []

    try:
        resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=min(n_recent, 50),
        ).execute()
    except HttpError:
        return []

    video_ids: List[str] = []
    for item in resp.get("items", []):
        vid = item.get("contentDetails", {}).get("videoId")
        if vid:
            video_ids.append(vid)

    if not video_ids:
        return []

    stats: List[VideoStat] = []
    for batch in chunks(video_ids, 50):
        try:
            resp = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(batch),
                maxResults=50,
            ).execute()
        except HttpError:
            continue

        for item in resp.get("items", []):
            title = item.get("snippet", {}).get("title", "")
            vc = safe_int(item.get("statistics", {}).get("viewCount"))
            stats.append(VideoStat(title=title, view_count=vc))

        time.sleep(0.05)

    return stats


def estimate_avg_views(video_stats: List[VideoStat]) -> Optional[int]:
    """Average views across last N videos (rounded)."""
    views = [v.view_count for v in video_stats if v.view_count is not None]
    if not views:
        return None
    return round(sum(views) / len(views))


# ----------------------------
# Output
# ----------------------------

def save_csv(path: str, rows: List[ChannelInfo], category: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "channel_name", "channel_url", "subscribers", "avg_views_last_n"])
        for r in rows:
            w.writerow([
                category,
                r.name,
                r.url,
                "" if r.subscribers is None else r.subscribers,
                "" if r.avg_views_last_n is None else r.avg_views_last_n,
            ])

    print(f"[OK] Saved {len(rows)} channels → {path}")


# ----------------------------
# Run: Category discovery
# ----------------------------

def process_category(
    youtube,
    category: str,
    keywords: List[str],
    title_terms: List[str],
    output_csv: str,
    pages_per_keyword: int,
    results_per_page: int,
    n_recent_videos: int,
    order: str,
    include_channel_search: bool,
    channel_pages_per_keyword: int,
    use_or_query: bool,
    min_hits: int,
    min_title_matches: int,
    min_subscribers: int,
    min_avg_views: int,
    min_view_to_sub_ratio: float,
    max_candidates_for_stats: int,
    target_count: int,
    ensure_target_count: bool,
) -> List[ScoredChannel]:
    hit_counts, _matched_keywords = discover_channel_ids(
        youtube,
        keywords=keywords,
        pages_per_keyword=pages_per_keyword,
        results_per_page=results_per_page,
        order=order,
        include_channel_search=include_channel_search,
        channel_pages_per_keyword=channel_pages_per_keyword,
        use_or_query=use_or_query,
    )
    print(f"[INFO] {category}: {len(hit_counts)} unique channels (before filters).")

    candidate_ids = select_candidate_ids(hit_counts, min_hits, max_candidates_for_stats)
    print(f"[INFO] {category}: {len(candidate_ids)} channels after min_hits={min_hits}.")
    if ensure_target_count and len(candidate_ids) < target_count and min_hits > 1:
        candidate_ids = select_candidate_ids(hit_counts, 1, max_candidates_for_stats)
        print(
            f"[INFO] {category}: relaxing min_hits to 1 to target {target_count} channels."
        )
    if ensure_target_count and len(candidate_ids) < target_count and max_candidates_for_stats:
        candidate_ids = select_candidate_ids(hit_counts, 1, 0)
        print(
            f"[INFO] {category}: expanding candidate pool to reach {target_count} channels."
        )

    meta = fetch_channel_metadata(youtube, list(candidate_ids))

    scored: List[ScoredChannel] = []
    for cid in candidate_ids:
        m = meta.get(cid)
        if not m:
            continue
        subs = m["subscribers"]
        uploads = m.get("uploads_playlist_id")
        if min_subscribers and (subs is None or subs < min_subscribers):
            continue

        video_stats = recent_video_stats_for_channel(
            youtube, uploads, n_recent=n_recent_videos
        )
        avg_views = estimate_avg_views(video_stats)
        title_matches = count_title_matches(video_stats, title_terms)

        if min_title_matches and title_matches < min_title_matches:
            continue
        if min_avg_views and (avg_views is None or avg_views < min_avg_views):
            continue
        if min_view_to_sub_ratio:
            if not subs or avg_views is None:
                continue
            if (avg_views / subs) < min_view_to_sub_ratio:
                continue

        scored.append(ScoredChannel(
            info=ChannelInfo(
                channel_id=cid,
                name=m["name"],
                url=f"https://www.youtube.com/channel/{cid}",
                subscribers=subs,
                avg_views_last_n=avg_views,
            ),
            hits=hit_counts.get(cid, 0),
            title_matches=title_matches,
        ))

    eligible = [
        s for s in scored
        if passes_quality_filters(s, min_subscribers, min_avg_views, min_view_to_sub_ratio)
    ]

    preferred = eligible
    if min_title_matches and not ensure_target_count:
        preferred = [s for s in eligible if s.title_matches >= min_title_matches]
    elif min_title_matches and ensure_target_count:
        filtered = [s for s in eligible if s.title_matches >= min_title_matches]
        if len(filtered) >= target_count:
            preferred = filtered
        else:
            print(
                f"[INFO] {category}: only {len(filtered)} channels meet min_title_matches="
                f"{min_title_matches}, filling with lower matches to reach {target_count}."
            )

    preferred.sort(key=relevance_sort_key, reverse=True)
    top_scored = preferred[:target_count] if target_count else preferred
    rows = [s.info for s in top_scored]

    print(f"[INFO] {category}: {len(rows)} channels after relevance/quality filters.")
    if target_count and len(rows) < target_count:
        print(
            f"[WARN] {category}: only {len(rows)} channels found; "
            f"increase pages_per_keyword or broaden keywords."
        )

    print(f"\nTop 10 for {category} (by relevance):")
    for s in top_scored[:10]:
        r = s.info
        print(
            f"- {r.name} | hits={s.hits} | title_hits={s.title_matches} | "
            f"subs={r.subscribers} | avg{n_recent_videos}={r.avg_views_last_n} | {r.url}"
        )

    save_csv(output_csv, rows, category=category)
    return top_scored


def main():
    youtube = youtube_client_from_env()

    categories = [
        {
            "name": "Design & Architecture",
            "keywords": [
                "interior design",
                "home renovation",
                "apartment makeover",
                "architecture design",
            ],
            "title_terms": [
                "interior",
                "design",
                "makeover",
                "renovation",
                "decor",
                "home",
                "apartment",
                "kitchen",
                "architecture",
                "floor plan",
            ],
        },
        {
            "name": "Editing & Short-Form Production Educators",
            "keywords": [
                "video editing tutorial",
                "short form editing",
                "capcut tutorial",
                "premiere pro tips",
            ],
            "title_terms": [
                "editing",
                "edit",
                "cut",
                "capcut",
                "premiere",
                "after effects",
                "color grade",
                "shorts",
                "reels",
                "timeline",
                "workflow",
            ],
        },
        {
            "name": "Explainer & Educational Storytelling Channels",
            "keywords": [
                "explainer video",
                "educational storytelling",
                "science explainers",
                "history explained",
            ],
            "title_terms": [
                "explainer",
                "explained",
                "story",
                "storytelling",
                "history",
                "science",
                "lesson",
                "how it works",
                "deep dive",
                "documentary",
                "breakdown",
            ],
        },
        {
            "name": "Fashion & Beauty Creators",
            "keywords": [
                "fashion haul",
                "beauty tutorial",
                "makeup routine",
                "style tips",
            ],
            "title_terms": [
                "fashion",
                "outfit",
                "style",
                "haul",
                "beauty",
                "makeup",
                "skincare",
                "routine",
                "lookbook",
                "try on",
                "grwm",
            ],
        },
        {
            "name": "Gaming Lore, Story & Narrative Creators",
            "keywords": [
                "game lore",
                "story recap",
                "game narrative",
                "lore explained",
            ],
            "title_terms": [
                "lore",
                "story",
                "narrative",
                "explained",
                "timeline",
                "recap",
                "ending",
                "theory",
                "analysis",
                "character",
                "world",
            ],
        },
        {
            "name": "Digital Art Creators",
            "keywords": [
                "digital art",
                "procreate tutorial",
                "photoshop painting",
                "illustration process",
            ],
            "title_terms": [
                "digital art",
                "procreate",
                "photoshop",
                "illustration",
                "speedpaint",
                "brush",
                "painting",
                "sketch",
                "render",
                "lineart",
            ],
        },
        {
            "name": "Sports (Analysis and Movement Visualization)",
            "keywords": [
                "sports analysis",
                "play breakdown",
                "movement mechanics",
                "game film",
            ],
            "title_terms": [
                "analysis",
                "breakdown",
                "film",
                "tactics",
                "movement",
                "mechanics",
                "play",
                "strategy",
                "visualization",
                "slow motion",
                "biomechanics",
            ],
        },
        {
            "name": "Indie Builders & Creators",
            "keywords": [
                "indie hacker",
                "build in public",
                "startup founder",
                "side project",
            ],
            "title_terms": [
                "indie",
                "build",
                "build in public",
                "startup",
                "founder",
                "ship",
                "launch",
                "saas",
                "side project",
                "maker",
            ],
        },
    ]

    # Tune these depending on how many channels you want.
    pages_per_keyword = 1         # low quota mode; increase if you need more candidates
    results_per_page = 50
    n_recent_videos = 10
    order = "relevance"           # or "viewCount" for bigger channels
    include_channel_search = False
    channel_pages_per_keyword = 0
    use_or_query = True

    # Precision / quality knobs (set to 0 to disable)
    min_hits = 1
    min_title_matches = 1
    min_subscribers = 0
    min_avg_views = 0
    min_view_to_sub_ratio = 0.0

    target_per_category = 50
    ensure_target_count = True
    max_candidates_for_stats = target_per_category * 3

    for cat in categories:
        category = cat["name"]
        output_csv = f"{slugify_category(category)}_channels.csv"
        process_category(
            youtube=youtube,
            category=category,
            keywords=cat["keywords"],
            title_terms=cat["title_terms"],
            output_csv=output_csv,
            pages_per_keyword=pages_per_keyword,
            results_per_page=results_per_page,
            n_recent_videos=n_recent_videos,
            order=order,
            include_channel_search=include_channel_search,
            channel_pages_per_keyword=channel_pages_per_keyword,
            use_or_query=use_or_query,
            min_hits=min_hits,
            min_title_matches=min_title_matches,
            min_subscribers=min_subscribers,
            min_avg_views=min_avg_views,
            min_view_to_sub_ratio=min_view_to_sub_ratio,
            max_candidates_for_stats=max_candidates_for_stats,
            target_count=target_per_category,
            ensure_target_count=ensure_target_count,
        )


if __name__ == "__main__":
    main()
