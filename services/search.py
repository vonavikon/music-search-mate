"""Metadata search on YouTube + SoundCloud via yt-dlp (NO downloading).

Data flow step 3: build the candidate pool from LLM#1's search strings.
extract_flat = fast (no full extract); we take id/title/channel/duration/views.
YouTube is the primary source, SoundCloud second (graceful degradation on failure).
"""
import asyncio
import logging
import os
import yt_dlp

log = logging.getLogger("services.search")

# Hard cap on synchronous yt-dlp calls run in a ThreadPoolExecutor.
# socket_timeout/retries bound only socket I/O, not the whole extract_info
# (pagination, backoff between retries, rate-limit): without wait_for a hung
# extract stalls the whole pipeline's asyncio.gather (incident 2026-08-31
# "Saloondria", 54 min). Values keep headroom over socket+retries.
# (2026-11-01) SC timeout tightened 90s -> 18s and SC moved to a fallback
# phase: a hung scsearch produced a latency tail exactly at the timeout
# (Langfuse: max youtube-search = 90.0s). SC search is now secondary: it
# runs only when the YouTube pool comes up short.
_SEARCH_TIMEOUT = 60.0      # ytsearch: socket_timeout=30, retries=3
_SEARCH_SC_TIMEOUT = 18.0   # scsearch (fallback phase): no longer on the main path
_ENRICH_TIMEOUT = 45.0      # full extract_info: socket_timeout=20, retries=2

# Target size of the unique-candidate pool (early exit from harvesting) and the
# shortfall threshold below which SoundCloud kicks in. Production measurement:
# raw=212 while only 1-3 get shown — an order of magnitude overproduction;
# 48 uniques comfortably cover the filter (20).
_POOL_TARGET = 48
_POOL_FLOOR = 12
_WINDOW = 3             # parallel yt extractions in the sliding window

_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": True,
    "skip_download": True,
    "noplaylist": True,
    "socket_timeout": 30,
    "retries": 3,
}


def _normalize(entry: dict) -> dict | None:
    vid = entry.get("id") or ""
    title = (entry.get("title") or "").strip()
    if not vid or not title:
        return None
    return {
        "video_id": vid,
        "title": title,
        "channel": (entry.get("uploader") or entry.get("channel") or entry.get("uploader_name") or "").strip(),
        "duration": entry.get("duration") or 0,
        "views": entry.get("view_count") or 0,
    }


def _extract_one(query: str, per_query: int) -> list[dict]:
    opts = {**_OPTS}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{per_query}:{query}", download=False)
    out = []
    for e in (info or {}).get("entries", []) or []:
        if e:
            n = _normalize(e)
            if n:
                out.append(n)
    return out


_SC_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": True,
    "skip_download": True,
    "noplaylist": True,
    "socket_timeout": 30,
    "retries": 5,  # SoundCloud rate-limiting is more aggressive
}


def _normalize_sc(entry: dict) -> dict | None:
    """SoundCloud entry → unified dict. webpage_url is required."""
    sid = entry.get("id") or ""
    title = (entry.get("title") or "").strip()
    url = (entry.get("webpage_url") or entry.get("url") or "").strip()
    if not sid or not title or not url:
        return None
    return {
        "video_id": sid,
        "title": title,
        "channel": (entry.get("uploader") or entry.get("channel") or "").strip(),
        "duration": entry.get("duration") or 0,
        "views": entry.get("view_count") or 0,
        "source": "soundcloud",
        "url": url,
    }


def _extract_one_sc(query: str, per_query: int) -> list[dict]:
    """SoundCloud search with retry on rate-limit (3 attempts, backoff 1.5s/3s/4.5s)."""
    import time
    opts = {**_SC_OPTS}
    info = None
    for attempt in range(3):
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(f"scsearch{per_query}:{query}", download=False)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
    out = []
    for e in (info or {}).get("entries", []) or []:
        if e:
            n = _normalize_sc(e)
            if n:
                out.append(n)
    return out


async def search_metadata(queries: list[str], per_query: int = 20, on_progress=None) -> list[dict]:
    """Parallel search across YouTube + SoundCloud (yt-first with early exit).

    Phase order: all YouTube queries run in parallel (as_completed); once the
    unique-candidate pool reaches _POOL_TARGET, the unfinished ones are cancelled
    (early exit: we used to churn 200+ results of which 1-3 get shown). SoundCloud
    kicks in ONLY if YouTube underfilled below _POOL_FLOOR — a rare case, and it
    was exactly a hung scsearch that produced a latency tail at the SC timeout.

    Dedup by normalized title+channel: a SoundCloud duplicate of an already taken
    YouTube track becomes its fallback_url; SC tracks without a duplicate become
    separate candidates. on_progress(stage, text) is an async progress callback."""
    loop = asyncio.get_running_loop()

    async def one_yt(q: str) -> list[dict]:
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _extract_one, q, per_query),
                timeout=_SEARCH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("ytsearch timeout for %r (>%.0fs)", q, _SEARCH_TIMEOUT)
            return []
        except Exception as e:
            log.warning("ytsearch failed for %r: %s", q, str(e)[:200])
            return []

    async def one_sc(q: str) -> list[dict]:
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _extract_one_sc, q, per_query),
                timeout=_SEARCH_SC_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("scsearch timeout for %r (>%.0fs)", q, _SEARCH_SC_TIMEOUT)
            return []
        except Exception as e:
            log.warning("scsearch failed for %r: %s", q, str(e)[:200])
            return []

    from services.match_score import _norm as _norm_for_dedup

    def _coerce(r: dict) -> dict:
        """Coerce a raw yt-dlp dict into the unified shape. In production
        _extract_one / _extract_one_sc already normalize; this guards mocks
        and off-nominal paths."""
        if not r.get("source"):
            # The SC extractor carries webpage_url; the YT extractor only the id, no url
            r["source"] = "soundcloud" if r.get("webpage_url") and not r.get("url") else "youtube"
        if r.get("video_id") is None and r.get("id"):
            r["video_id"] = r["id"]
        if not r.get("channel") and r.get("uploader"):
            r["channel"] = r["uploader"]
        if not r.get("url") and r.get("webpage_url"):
            r["url"] = r["webpage_url"]
        return r

    def _key(r: dict) -> str:
        # channel comes from _normalize/_normalize_sc; for a raw yt-dlp dict
        # (uploader) and test mocks, check both keys.
        ch = r.get("channel") or r.get("uploader") or ""
        return _norm_for_dedup(r.get("title", "")) + "|" + _norm_for_dedup(ch)

    seen: set[str] = set()
    merged: list[dict] = []
    sc_by_key: dict[str, dict] = {}

    # --- phase 1: YouTube, sliding window + early exit once the pool fills ---
    # A window of _WINDOW parallel extractions: don't start all 8 queries at
    # once — feed more in only while the pool is short of the target. This
    # cuts both latency (no waiting on the slowest) and the number of yt-dlp
    # calls (measured overproduction: raw=212 to show 1-3).
    queue = list(queries)
    next_i = 0
    inflight: set = set()

    def _merge_yt(batch):
        for r in (batch or []):
            if not r:
                continue
            r = _coerce(r)
            if r.get("source", "youtube") != "youtube":
                continue
            key = _key(r)
            if key in seen:
                continue
            seen.add(key)
            r["source"] = "youtube"
            r["url"] = f"https://www.youtube.com/watch?v={r['video_id']}"
            merged.append(r)

    try:
        while True:
            while (len(inflight) < _WINDOW and next_i < len(queue)
                   and len(merged) < _POOL_TARGET):
                inflight.add(asyncio.ensure_future(one_yt(queue[next_i])))
                next_i += 1
            if not inflight or len(merged) >= _POOL_TARGET:
                break
            done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                try:
                    _merge_yt(fut.result())
                except Exception:
                    continue
    except Exception as e:
        log.warning("ytsearch phase failed: %s", str(e)[:150])
    finally:
        for t in inflight:
            if not t.done():
                t.cancel()

    # --- phase 2: SoundCloud only on shortfall (usually skipped) ---
    if len(merged) < _POOL_FLOOR:
        log.info("search pool underfilled (yt=%d < %d) — trying soundcloud",
                 len(merged), _POOL_FLOOR)
        sc_tasks = {asyncio.ensure_future(one_sc(q)): q for q in queries}
        try:
            for fut in asyncio.as_completed(sc_tasks, timeout=_SEARCH_SC_TIMEOUT + 5.0):
                try:
                    batch = await fut
                except Exception:
                    continue
                for r in (batch or []):
                    if not r:
                        continue
                    r = _coerce(r)
                    if r.get("source", "youtube") == "youtube":
                        continue
                    key = _key(r)
                    if key not in sc_by_key:
                        sc_by_key[key] = r
                if len(merged) + len(sc_by_key) >= _POOL_TARGET:
                    break
        except asyncio.TimeoutError:
            pass
        finally:
            for t in sc_tasks:
                if not t.done():
                    t.cancel()

    # 3. An SC duplicate of a YouTube track -> fallback_url (not a separate result).
    for r in merged:
        sc = sc_by_key.pop(_key(r), None)
        if sc:
            r["fallback_url"] = sc["url"]

    # 4. Remaining SoundCloud tracks (no YouTube duplicate) — standalone results.
    for r in sc_by_key.values():
        merged.append(r)

    if on_progress:
        try:
            await on_progress("search", f"Ищу… нашёл {len(merged)}")
        except Exception:
            pass

    return merged


async def enrich_candidates(
    candidates: list[dict],
    metadata_svc,  # MetadataService or None
    top_n: int = 5,
) -> list[dict]:
    """For the top-N candidates: full extract_info + Last.fm → enriched metadata.
    The rest are returned unchanged. Does not mutate the original."""
    if not candidates:
        return candidates

    loop = asyncio.get_running_loop()

    async def enrich_one(c: dict) -> dict:
        c = dict(c)  # don't mutate the original
        url = c.get("url") or f"https://www.youtube.com/watch?v={c['video_id']}"

        # layer 1: yt-dlp extract_info (no download) → artist/track/album/year
        try:
            info = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: _extract_info_no_dl(url)),
                timeout=_ENRICH_TIMEOUT,
            )
            if info:
                c["canonical_artist"] = (info.get("artist") or "").strip()
                c["canonical_track"] = (info.get("track") or info.get("title") or "").strip()
                c["canonical_album"] = (info.get("album") or "").strip()
                release_date = info.get("release_date") or info.get("upload_date") or ""
                c["canonical_year"] = str(release_date)[:4] if release_date and len(str(release_date)) >= 4 else ""
                c["canonical_description"] = (info.get("description") or "")[:300]
                c["categories"] = info.get("categories") or []
        except asyncio.TimeoutError:
            log.debug("enrich extract_info timeout for %s (>%.0fs)", c.get("video_id"), _ENRICH_TIMEOUT)
        except Exception as e:
            log.debug("enrich extract_info failed for %s: %s", c.get("video_id"), str(e)[:100])

        # layer 2: Last.fm (if the service and key are present)
        if metadata_svc and metadata_svc._lastfm_key:
            artist = c.get("canonical_artist", "") or c.get("channel", "")
            track = c.get("canonical_track", "") or c.get("title", "")
            if artist and track and artist != "Unknown Artist":
                try:
                    lf = await metadata_svc._lastfm_getinfo(artist, track)
                    if lf:
                        c["canonical_artist"] = lf.get("artist") or c.get("canonical_artist", "")
                        c["canonical_track"] = lf.get("track") or c.get("canonical_track", "")
                        c["canonical_genre"] = lf.get("genre") or ""
                        c["canonical_bio"] = lf.get("bio") or ""
                except Exception:
                    pass
                # artist-level fallback for genre
                if not c.get("canonical_genre") and artist:
                    try:
                        c["canonical_genre"] = await metadata_svc._artist_toptags(artist)
                    except Exception:
                        pass

        return c

    # enrich the top-N in parallel
    enriched = list(candidates)
    if top_n > 0:
        top_tasks = [enrich_one(c) for c in enriched[:top_n]]
        top_results = await asyncio.gather(*top_tasks)
        enriched[:top_n] = top_results

    return enriched


def _yt_cookiefile() -> str | None:
    """Cookies from a logged-in session pass YouTube's datacenter-IP bot check
    (a LOGIN_REQUIRED wave since 2026-10-06). The file is mounted from
    ./secrets; no file -> None, previous behavior."""
    p = os.getenv("YTDLP_COOKIEFILE")
    return p if p and os.path.exists(p) else None


def _extract_info_no_dl(url: str) -> dict | None:
    """Full extract_info without downloading. Returns the info dict or None."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 20,
        "cookiefile": _yt_cookiefile(),
        "retries": 2,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)
