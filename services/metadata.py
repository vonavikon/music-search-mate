"""Canonical track metadata cascade.

Three sources, cheapest first:
  1. yt-dlp info dict      - artist/track/album/release_date/thumbnails.
                             Labels fill these via YouTube "Music in this video" (Content ID).
                             Free, 0 requests, already fetched at download time.
  2. Last.fm track.getInfo - canonical artist, year, genre, bio (the "what/where from").
                             Fills yt-dlp gaps for amateur uploads.
                             Graceful: skipped when LASTFM_API_KEY is empty.
  3. iTunes Search         - 600x600 artwork (a real album cover, not a music-video frame).
                             No key needed. Fallback to the yt-dlp thumbnail.

TG-agnostic. Any external API failure never breaks the cascade: fall back to the
layer below.
"""
import logging
import re
import time
from dataclasses import dataclass

import httpx

from config import TEMP_DIR

_log = logging.getLogger("bot.metadata")
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


@dataclass
class MetaOut:
    artist: str
    track: str
    album: str = ""
    year: str = ""
    genre: str = ""
    bio: str = ""
    bio_about: str = ""  # "" | "track" | "artist": whose bio it is (e.g. "About the artist:")
    artwork_url: str = ""  # iTunes artwork URL (for send_photo directly)
    artwork_path: str | None = None  # path to the downloaded artwork or None

    @property
    def has_artwork(self) -> bool:
        return bool(self.artwork_path)


def _clean_channel(ch: str) -> str:
    """Fallback artist from the channel name when yt-dlp gave no canonical one. Strips '- Topic'."""
    if not ch:
        return "Unknown Artist"
    ch = re.sub(r"\s*[-–—]\s*topic\s*$", "", ch, flags=re.I)
    ch = ch.strip(" -–—|")
    return ch or "Unknown Artist"


def _year_from_date(d: str | None) -> str:
    """yt-dlp 'YYYYMMDD' release_date/upload_date -> 'YYYY'."""
    if not d or not str(d).isdigit() or len(str(d)) < 4:
        return ""
    return str(d)[:4]


# Clean a raw YouTube title down to canonical. Strips tails that carry upload
# metadata rather than the track name: OST/soundtrack, feat./ft., remaster,
# official video, lyrics, HD/4K, from a movie. Applied BEFORE Last.fm/iTunes —
# a clean title gives better recall (junk like "... (OST Taxi)" misses the exact
# track). Only matched inside brackets/pipes: "by X" outside brackets can be part
# of the name (DJ-scratch and such).
_NOISE_STRIP = re.compile(
    r"\s*[\(\[\|]\s*(?:official\s*(?:video|audio|music\s*video|lyric\s*video)|"
    r"music\s*video|hd|4k|hq|with\s*lyrics?|lyrics?|audio|video|"
    r"remastered(?:\s*\d{4})?|re-?recorded|"
    r"ost|o\.?s\.?t\.?|soundtrack|original\s+motion\s+picture\s+soundtrack|"
    r"из\s+(?:к/ф|т/с|фильма|сериала|мультфильма|игры)|main\s+theme|theme\s+song|"
    r"original\s+mix|extended\s+mix|radio\s+edit|"
    r"feat\.?\s+[^)\]\|]+|ft\.?\s+[^)\]\|]+|"
    r"cover(?:\s+version)?|karaoke|tribute)\s*[\)\]\|]",
    re.I,
)
_PAREN_TRAILING = re.compile(r"\s*[\(\[\|]\s*[\)\]\|]\s*$")  # empty brackets left after the strip


def _clean_raw_title(t: str) -> str:
    """raw YouTube title -> cleaned: strips noisy bracketed tails, empty brackets,
    edge separators. Idempotent — a re-run doesn't change a clean title."""
    t = _NOISE_STRIP.sub("", t or "")
    t = _PAREN_TRAILING.sub("", t)
    return t.strip(" -–—|")


# Last.fm behavioral/service tags that reach the top via scrobbling - not genres.
# Without the filter the card shows "rock, seen live, favorites" instead of "rock, 80s".
_TAG_DENYLIST = {
    "seen live", "seen them live", "want to see live", "favorites", "favourite",
    "favorite", "favourites", "my favorites", "my favourites", "owned", "owning",
    "downloaded", "download", "spotify", "good music", "loved", "love", "awesome",
    "under 2000 listeners", "all time favs", "my music",
}


# iTunes pseudo-artists: if artistName is one of these, do NOT overwrite the
# canonical artist — these are compilations/soundtracks; overwriting would yield
# "Various Artists" instead of the real one.
_GENERIC_ARTIST = {
    "various artists", "various", "various artist", "soundtrack",
    "original motion picture soundtrack", "original soundtrack",
    "original cast", "original cast recording", "cast", "compilation",
    "speech", "audiobook", "unknown", "unknown artist", "no artist",
}


# Strip version suffixes BEFORE the iTunes artwork search — the case
# "Diana Ross Upside Down (slowed reverb)": with the suffix iTunes matches foreign
# slowed/nightcore compilations and the card gets a "strange" artwork.
_STRIP_VERSION_RE = re.compile(
    r"\s*[\(\[\|]\s*(?:slowed|reverb|sped\s*up|speed\s*up|nightcore|8d)[^\)\]\|]*[\)\]\|]"
    r"|\s+(?:slowed|sped\s*up|nightcore)(?:\s*\+\s*|\s+)reverb\s*$",
    re.I)


def _strip_versions(track: str) -> str:
    return _STRIP_VERSION_RE.sub("", track or "").strip(" -\u2013\u2014|()[]")


def _tag_key(s: str) -> str:
    """Normalize a tag/artist name for comparison: casefold, punctuation out,
    & -> and, whitespace collapsed. For exact (not contains) comparison."""
    s = s.casefold().strip()
    s = s.replace("&", "and")  # BEFORE punctuation: "&" becomes the word "and", not a space
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _gate_track(lf_name: str, requested: str) -> str:
    """track.getInfo can return a namesake track of another artist. Overwrite the
    canonical name from Last.fm only if it is close to the requested one (exact
    match, or the requested name is a substring of the Last.fm name at length
    >= 8), otherwise keep the requested cleaned title. Protects against silently
    swapping the title for a foreign track. Short names (len < 8) are often
    namesakes (Creep, Song) — exact match only, no contains."""
    lf_key = _tag_key(lf_name)
    req_key = _tag_key(requested)
    if not req_key or not lf_key:
        return lf_name
    if lf_key == req_key:
        return lf_name
    if len(req_key) >= 8 and req_key in lf_key:
        return lf_name  # Last.fm refined it with a suffix (Album Version / feat.), the root matches
    return requested  # names diverged — a foreign namesake track, don't substitute


_FEAT_RE = re.compile(r"\bfeat\.?\b|\bft\.?\b", re.I)


def _is_pseudo_artist(artist: str) -> bool:
    """Pseudo-artist for the artist-level fallback: a compilation/Various Artists
    or a feat-aggregate ("A feat. B"). genre/bio are garbage for those — skip layer 2.5."""
    key = _tag_key(artist)
    if not key or key in _GENERIC_ARTIST:
        return True
    return bool(_FEAT_RE.search(artist))


def _gate_artist(new_artist: str, requested: str) -> str:
    """Protect the canonical artist from substitution by a foreign artist (feat/&/
    short names). Overwrite only on closeness: exact match, or one name is a
    substring of the other at length >= 8. Otherwise keep the requested one —
    autocorrect or a namesake artist may have led it astray. Short names (len < 8)
    are more often namesakes — exact only."""
    new_key = _tag_key(new_artist)
    req_key = _tag_key(requested)
    if not req_key or not new_key:
        return new_artist
    if new_key == req_key:
        return new_artist
    if len(req_key) >= 8 and (req_key in new_key or new_key in req_key):
        return new_artist
    return requested


_ALBUM_EDITION_RE = re.compile(
    r"\s*[\(\[\|]?\s*"
    r"(?:\d{4}[\s\-]*)?"
    r"(?:"
    r"deluxe|expanded|remaster(?:ed)?|re-?master(?:ed)?|"
    r"(?:\d+(?:st|nd|rd|th)?\s+)?anniversary|"
    r"bonus(?:\s+track(?:s)?)?|special(?:\s+edition)?|"
    r"collector'?s?(?:\s+edition)?|limited(?:\s+edition)?|reissue|re-?issue"
    r")"
    r"(?:\s*\d{4})?"
    r"\s*(?:edition|version)?"
    r"\s*[\)\]\|]?\s*$",
    re.I,
)


def _normalize_album(album: str) -> str:
    """Cut the edition tail off an album name ("The Wall (Deluxe Edition)"
    -> "The Wall"). For a clean tag and for scoring candidates from different
    sources. Idempotent — a clean album is unchanged."""
    a = (album or "").strip()
    if not a:
        return a
    out = _ALBUM_EDITION_RE.sub("", a).strip(" -–—|")
    return out or a


def _pick_album(candidates: list[str]) -> str:
    """Pick the best album from candidates of different sources. Each is
    normalized (_normalize_album); the shortest normalized one wins (edition
    stripped), ties broken by order. A single candidate is just normalized.
    Empty -> empty."""
    cands = [c.strip() for c in candidates if c and c.strip()]
    if not cands:
        return ""
    best = min(cands, key=lambda c: len(_normalize_album(c)))
    return _normalize_album(best)


def _parse_tags(raw, artist: str = "") -> str:
    """Last.fm toptags (dict | list | None) -> a genre string. Filters junk tags
    and the tag equal to the artist name (exact match - otherwise it would break
    'Jackson 5' and such); takes top-3, lowercased for a uniform card."""
    if not raw:
        return ""
    if isinstance(raw, dict):
        raw = [raw]
    art_key = _tag_key(artist) if artist else ""
    names = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        name = (t.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in _TAG_DENYLIST or key.startswith("seen "):
            continue
        if art_key and _tag_key(name) == art_key:
            continue  # tag = artist name (a marker), not a genre
        names.append(name)
    return ", ".join(names[:3]).lower() if names else ""


def _itunes_match(artist_key: str, track_key: str, res: dict) -> bool:
    """Match-gate for iTunes: confirms the result is the very track, not a popular
    miss. artist — exact match (normalized), track — exact or contains at
    sufficient length (the canonical name is shorter than the cleaned YouTube
    title), not tribute/cover/karaoke/instrumental. Without the gate iTunes
    substitutes metadata silently: it returns a popular partial-query match that
    looks official but is foreign."""
    an = _tag_key(res.get("artistName", ""))
    tn = _tag_key(res.get("trackName", ""))
    cn = _tag_key((res.get("collectionName") or "") + " " + (res.get("trackName") or ""))
    if any(b in cn for b in ("tribute", "cover", "karaoke")):
        return False
    artist_ok = bool(an and an == artist_key)
    track_ok = bool(tn and (tn == track_key or (len(tn) >= 8 and tn in track_key)))
    return artist_ok and track_ok


def _titlecase_ru_en(s: str) -> str:
    """Careful title-case: capitalize every word except function words and words
    that already contain uppercase (acronyms/names untouched). Applied only to
    strings that are entirely lowercase — a sign of a junk title."""
    minor = {"of", "the", "a", "an", "and", "in", "on", "at", "to", "for",
             "feat", "ft", "vs", "или", "и", "на", "в", "с", "для"}
    out = []
    for i, w in enumerate(s.split(" ")):
        if w.islower() and (i == 0 or w not in minor):
            out.append(w[:1].upper() + w[1:])
        else:
            out.append(w)
    return " ".join(out)


def pretty_title(raw: str) -> str:
    """Clean YouTube junk from the displayed title when the canonical source didn't
    confirm it (prod cases: "jerry lee lewis great balls of fire", "Chuck Berry,
    Chuck Berry", trailing spaces): underscores/plus -> space, whitespace collapse,
    separator-tail trimming, comma-duplicate removal, title-case for all-lowercase
    strings."""
    s = (raw or "").strip()
    if not s:
        return s
    s = re.sub(r"[_+]+", " ", s)
    # markdown junk from LLM/sonar (**) doesn't render in TG (HTML)
    s = re.sub(r"\*{1,2}|_{1,2}|`{1,3}", "", s)
    # video-marker brackets carry no meaning in a track title (prod acceptance:
    # the tag "Iam - Petit Frère (Clip Officiel)")
    s = re.sub(r"\s*\((official\s*)?(video|audio|clip\s*officiel|clip|lyrics?|"
               r"visualizer|hd|4k|hq|remaster(ed)?(\s*\d{4})?|hd\s+remaster(ed)?(\s*\d{4})?)\)",
               "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip(" -|\u2013\u2014\u00b7")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        s = parts[0]  # "Chuck Berry, Chuck Berry" -> "Chuck Berry"
    has_upper = any(c.isupper() for c in s)
    if not has_upper:
        s = _titlecase_ru_en(s)
    return s


class _TTLCache:
    """An in-memory TTL cache for Last.fm responses. Popular references repeat
    ("something like Queen" gets asked dozens of times), and similar/track.search/
    getinfo for one key return the same thing for weeks. A network call = 0.3-1.5s
    latency, the cache = 0. Naive eviction: on overflow it clears entirely (simple
    and sufficient: the working key set is far smaller than maxsize)."""

    __slots__ = ("ttl", "maxsize", "_d")

    def __init__(self, ttl: float = 24 * 3600, maxsize: int = 512):
        self.ttl = ttl
        self.maxsize = maxsize
        self._d: dict = {}

    def get(self, key):
        hit = self._d.get(key)
        if not hit:
            return None, False
        ts, val = hit
        if time.time() - ts > self.ttl:
            self._d.pop(key, None)
            return None, False
        return val, True

    def put(self, key, val) -> None:
        if len(self._d) >= self.maxsize:
            self._d.clear()
        self._d[key] = (time.time(), val)


_SENTINEL = object()


class MetadataService:
    def __init__(self, lastfm_api_key: str = ""):
        self._lastfm_key = (lastfm_api_key or "").strip()
        self._http = httpx.AsyncClient(timeout=_TIMEOUT)
        self._cache = _TTLCache()  # Last.fm responses cached for a day

    async def aclose(self):
        await self._http.aclose()

    async def _artist_from_title(self, raw_title: str) -> str:
        """"Metrik - Gravity" -> "Metrik", with the artist verified via artist.getInfo
        (canonical name; junk left parts ("Various Artists", "Best of") are
        filtered by _is_pseudo_artist). Empty = no guess.
        The "Track by Artist" form splits with the same validation (case:
        "Freestate by Depeche Mode" -> the Dameonryan channel became the artist)."""
        m_by = re.match(r"^(.{2,60}?)\s+by\s+([A-ZА-ЯЁ][\w&.'’\- ]{1,40})\s*$",
                        raw_title)
        if " - " in raw_title:
            left, right = raw_title.split(" - ", 1)
            left, right = left.strip(), right.strip()
        elif m_by:
            left, right = m_by.group(2), m_by.group(1)  # "by" form: artist on the right
        else:
            return ""
        # the right part must look like a track name, the left — no longer than a
        # typical artist (protection against "Top 100 - Track Name" splits)
        if not left or len(left) < 2 or len(left) > 40 or len(right) < 2:
            return ""
        if _is_pseudo_artist(left):
            return ""
        try:
            info = await self._artist_getinfo(left)
            if info and info.get("artist"):
                return info["artist"]
        except Exception:
            pass
        return ""

    async def resolve(self, info: dict, video_id: str) -> MetaOut:
        """Cascade: yt-dlp -> Last.fm -> iTunes artwork. Never raises."""
        # --- layer 1: yt-dlp info dict ---
        artist = (info.get("artist") or "").strip()
        artist_from_content_id = bool(artist)  # Content ID gave an artist; False = channel fallback (unreliable)
        raw_track = (info.get("track") or "").strip() or (info.get("title") or "").strip()
        track = _clean_raw_title(raw_track)  # strip OST/feat/remastered tails before Last.fm/iTunes
        album = (info.get("album") or "").strip()
        album_candidates = [album] if album else []
        year = _year_from_date(info.get("release_date") or info.get("upload_date"))
        if not artist:
            artist = _clean_channel(info.get("channel") or info.get("uploader") or "")

        # The answer may sit in the title itself ("Metrik - Gravity") when the
        # uploader channel stands in as the artist (e.g. "UKF Drum & Bass"). Take
        # the artist from the "X - Y" split ONLY when yt-dlp gave no Content ID
        # artist (the channel is unreliable), verified via artist.getInfo: exists
        # and not a pseudo-artist. No substitution gate needed — we replace the
        # worse (channel) with a verified value, not the other way round.
        genre = bio = ""
        bio_about = ""
        sources = ["ytdlp"]  # trace: which layers contributed canonical data

        if not artist_from_content_id and self._lastfm_key:
            guessed = await self._artist_from_title(raw_track or "")
            if guessed:
                artist = guessed
                sources.append("title-split")

        # --- layer 2: Last.fm (if key present) ---
        if self._lastfm_key and artist and track and artist != "Unknown Artist":
            try:
                lf = await self._lastfm_getinfo(artist, track)
                if lf:
                    # Last.fm canonical takes priority (label tagging is more precise).
                    # The Content ID artist is protected by a gate: don't substitute it
                    # with a foreign one (feat/&/short names).
                    if lf.get("artist"):
                        artist = _gate_artist(lf["artist"], artist) if artist_from_content_id else lf["artist"]
                    track = lf.get("track") or track
                    if lf.get("album"):
                        album_candidates.append(lf["album"].strip())
                    year = year or lf.get("year") or ""
                    genre = lf.get("genre") or ""
                    bio = lf.get("bio") or ""
                    bio_about = "track" if bio else bio_about
                    sources.append("lastfm")
            except Exception as e:
                _log.debug("lastfm skipped: %s", str(e)[:100])

        # --- layer 2.5: artist-level fallback (genre/bio when track-level is empty).
        # The artist has accumulated data even when the specific track is sparse.
        # getInfo goes first: autocorrect normalizes the name, the canonical artist
        # improves artwork (layer 3) and the toptags request. toptags - by the
        # normalized name. ---
        if (self._lastfm_key and artist and artist != "Unknown Artist"
                and not _is_pseudo_artist(artist) and (not genre or not bio)):
            try:
                af = await self._artist_getinfo(artist)
                if af:
                    if af.get("artist"):
                        artist = _gate_artist(af["artist"], artist) if artist_from_content_id else af["artist"]
                    if not bio:
                        bio = af.get("bio") or ""
                        bio_about = "artist" if bio else bio_about
                had_genre = bool(genre)
                if not genre:
                    genre = await self._artist_toptags(artist)
                if af or (not had_genre and genre):
                    sources.append("lastfm-artist")
            except Exception as e:
                _log.debug("artist-level skipped: %s", str(e)[:100])

        # --- layer 3: iTunes — artwork always + canonical fields behind a match-gate ---
        # Without the gate iTunes substitutes metadata silently: it returns a popular
        # partial-query match that looks official but is foreign. gate = exact artist
        # + track match (normalized), not tribute/cover. Canonical fields take
        # priority over yt-dlp/Last.fm (iTunes label tagging is more precise), but
        # only when confirmed. Artwork fallback to the yt-dlp thumbnail — at the
        # handler level.
        artwork_path = None
        artwork_url = ""
        if artist and track and artist != "Unknown Artist":
            try:
                it = await self._itunes_resolve(artist, track, video_id)
                artwork_path = it.get("artwork_path")
                artwork_url = it.get("artwork_url") or ""
                if it.get("track"):
                    track = it["track"]
                    sources.append("itunes")  # gate passed — canonical fields confirmed
                elif artwork_path:
                    sources.append("art")     # artwork only (top-1 fallback), gate missed
                # Overwrite the artist with the iTunes canonical only if it's not a
                # pseudo-artist (Various Artists / Soundtrack) — otherwise we'd erase
                # the real performer.
                if it.get("artist") and _tag_key(it["artist"]) not in _GENERIC_ARTIST:
                    artist = it["artist"]
                if it.get("album"):
                    album_candidates.append(it["album"].strip())
                year = year or it.get("year") or ""
            except Exception as e:
                _log.debug("itunes skipped: %s", str(e)[:100])

        # Album candidate scoring: normalize and pick the best across sources.
        album = _pick_album(album_candidates)

        # observability: the final cascade — which fields and from which layers.
        _log.info(
            "meta %s | artist=%s track=%s album=%s year=%s genre=%s art=%s | %s",
            video_id, artist[:30], track[:40], (album or "-")[:25], year or "-",
            (genre[:25] if genre else "-"), "y" if artwork_path else "n",
            "+".join(sources),
        )
        # acceptance: the track tag must not repeat the artist ("Iam - Petit Frère…")
        if artist and track.lower().startswith(artist.lower() + " - "):
            track = track[len(artist) + 3:]
        return MetaOut(
            artist=pretty_title(artist), track=pretty_title(track), album=album, year=year,
            bio_about=bio_about,
            genre=genre, bio=bio, artwork_path=artwork_path, artwork_url=artwork_url,
        )

    async def _lastfm_getinfo(self, artist: str, track: str) -> dict | None:
        """track.getInfo: canonical artist, album, year, genre, bio. None on a miss."""
        key = ("getinfo", (artist or "").strip().lower(), (track or "").strip().lower())
        cached, hit = self._cache.get(key)
        if hit:
            return cached
        result = await self._lastfm_getinfo_uncached(artist, track)
        self._cache.put(key, result)
        return result

    async def _lastfm_getinfo_uncached(self, artist: str, track: str) -> dict | None:
        r = await self._http.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "track.getInfo",
                "api_key": self._lastfm_key,
                "artist": artist,
                "track": track,
                "autocorrect": "0",
                "format": "json",
            },
        )
        if r.status_code != 200:
            return None
        data = r.json()
        td = data.get("track") or {}
        if not td or "error" in data:
            return None
        out = {}
        if td.get("name"):
            # namesake gate: don't substitute the requested title with a foreign track
            out["track"] = _gate_track(td["name"].strip(), track)
        a = td.get("artist") or {}
        if isinstance(a, dict) and a.get("name"):
            out["artist"] = a["name"].strip()
        alb = td.get("album") or {}
        if isinstance(alb, dict):
            if alb.get("title"):
                out["album"] = alb["title"].strip()
            # wiki "published" is more precise than the album year, but take it only if yt-dlp gave none
        wiki = td.get("wiki") or {}
        # Last.fm wiki "published" = the article's publication/update date, NOT the
        # release year (Billie Jean returns 2008 instead of 1983, Tame Impala 2016
        # instead of 2015). A systematically wrong source - not used. The year comes
        # only from yt-dlp release_date (layer 1, Content ID). summary - a valid
        # source for bio.
        if isinstance(wiki, dict) and wiki.get("summary"):
            # summary ends with an <a href=...> link attribute — cut it off
            out["bio"] = wiki["summary"].split("<a")[0].strip()
        genre = _parse_tags(td.get("toptags", {}).get("tag"), artist)
        if genre:
            out["genre"] = genre
        return out or None

    async def _artist_toptags(self, artist: str) -> str:
        """artist.getTopTags: the artist's genre. Richer and more reliable than
        track tags - an artist almost always has tags, a specific track often
        has none."""
        key = ("toptags", (artist or "").strip().lower())
        cached, hit = self._cache.get(key)
        if hit:
            return cached
        result = await self._artist_toptags_uncached(artist)
        self._cache.put(key, result)
        return result

    async def _artist_toptags_uncached(self, artist: str) -> str:
        r = await self._http.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "artist.getTopTags",
                "api_key": self._lastfm_key,
                "artist": artist,
                "autocorrect": "0",
                "format": "json",
            },
        )
        if r.status_code != 200:
            return ""
        data = r.json()
        if "error" in data:
            return ""
        return _parse_tags(data.get("toptags", {}).get("tag"), artist)

    async def _artist_getinfo(self, artist: str) -> dict | None:
        """artist.getInfo: canonical artist (name) + bio (summary). The canonical
        name is needed even when bio is empty - autocorrect may have fixed a typo
        in the artist name, and the normalized name improves layer 3 (iTunes
        artwork) and toptags. None on a complete miss. Similar artists are
        fetched separately via similar_artists() (sonic mode)."""
        key = ("artistgetinfo", (artist or "").strip().lower())
        cached, hit = self._cache.get(key)
        if hit:
            return cached
        result = await self._artist_getinfo_uncached(artist)
        self._cache.put(key, result)
        return result

    async def _artist_getinfo_uncached(self, artist: str) -> dict | None:
        r = await self._http.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "artist.getInfo",
                "api_key": self._lastfm_key,
                "artist": artist,
                "autocorrect": "0",
                "format": "json",
            },
        )
        if r.status_code != 200:
            return None
        data = r.json()
        ad = data.get("artist") or {}
        if not ad or "error" in data:
            return None
        out = {}
        if ad.get("name"):
            out["artist"] = ad["name"].strip()
        bio_obj = ad.get("bio") or {}
        if isinstance(bio_obj, dict) and bio_obj.get("summary"):
            out["bio"] = bio_obj["summary"].split("<a")[0].strip()
        return out or None

    async def similar_artists(self, artist: str, limit: int = 10) -> list[str]:
        """artist.getSimilar: the artist's empirical sonic neighbors. Last.fm builds
        similarity from user scrobbles, not genre labels, so it's a "sounds like"
        signal, not "same genre". [] on a miss or without a key. Used to adjust
        ranking for intent=similar. The response is cached for a day."""
        if not self._lastfm_key or not (artist or "").strip():
            return []
        key = ("similar", artist.strip().lower(), limit)
        cached, hit = self._cache.get(key)
        if hit:
            return cached
        result = await self._similar_artists_uncached(artist, limit)
        self._cache.put(key, result)
        return result

    async def _similar_artists_uncached(self, artist: str, limit: int) -> list[str]:
        try:
            r = await self._http.get(
                "https://ws.audioscrobbler.com/2.0/",
                params={
                    "method": "artist.getSimilar",
                    "api_key": self._lastfm_key,
                    "artist": artist,
                    "autocorrect": "0",
                    "limit": str(limit),
                    "format": "json",
                },
            )
            if r.status_code != 200:
                return []
            data = r.json()
            if "error" in data:
                return []
            sims = data.get("similarartists", {}).get("artist") or []
            return [s["name"].strip() for s in sims
                    if isinstance(s, dict) and s.get("name")]
        except Exception as e:
            _log.debug("getSimilar skipped: %s", str(e)[:100])
            return []

    async def track_search(self, track: str, limit: int = 8) -> list[dict]:
        """track.search: an empirical resolve of a track NAME into real artists.
        The key similar-mode case: the user names a track plus non-music context
        ("Jennifer Connelly the night has a thousand eyes") — LLM#1 puts the
        actress into reference.artist and getSimilar silently returns nothing for
        her. track.search by name returns the track's real artists with listeners.
        [{artist, listeners}] sorted by listeners descending; [] on a miss/without
        a key. The response is cached for a day."""
        if not self._lastfm_key or not (track or "").strip():
            return []
        key = ("tracksearch", track.strip().lower(), limit)
        cached, hit = self._cache.get(key)
        if hit:
            return cached
        result = await self._track_search_uncached(track, limit)
        self._cache.put(key, result)
        return result

    async def _track_search_uncached(self, track: str, limit: int) -> list[dict]:
        try:
            r = await self._http.get(
                "https://ws.audioscrobbler.com/2.0/",
                params={
                    "method": "track.search",
                    "api_key": self._lastfm_key,
                    "track": track.strip(),
                    "limit": str(limit),
                    "format": "json",
                },
            )
            if r.status_code != 200:
                return []
            data = r.json()
            if "error" in data:
                return []
            matches = data.get("results", {}).get("trackmatches", {}).get("track") or []
            out = []
            for m in matches:
                if not isinstance(m, dict) or not (m.get("artist") or "").strip():
                    continue
                try:
                    listeners = int(m.get("listeners") or 0)
                except (TypeError, ValueError):
                    listeners = 0
                out.append({"artist": m["artist"].strip(), "listeners": listeners})
            out.sort(key=lambda x: x["listeners"], reverse=True)
            return out
        except Exception as e:
            _log.debug("track.search skipped: %s", str(e)[:100])
            return []

    async def refetch_artwork(self, artist: str, track: str, video_id: str = "") -> str:
        """Artwork only, by the canonical artist/track pair — for cache hits of old
        file_cache records (created before artwork_url existed, it's empty).
        Returns the artwork URL or ""."""
        try:
            it = await self._itunes_resolve(artist, track, video_id)
            return it.get("artwork_url") or ""
        except Exception as e:
            _log.debug("refetch_artwork skipped: %s", str(e)[:100])
            return ""

    async def _itunes_resolve(self, artist: str, track: str, video_id: str) -> dict:
        """iTunes search -> artwork always (top-1 fallback) + canonical fields
        behind a match-gate. Returns dict {artwork_path, track, artist, album,
        year}; canonical fields are empty if the gate didn't pass (then artwork
        only, metadata is not substituted). limit=5: the gate checks several
        results, the popular-but-wrong top-1 is often lower."""
        track = _strip_versions(track) or track  # no version suffixes — no garbage matches
        r = await self._http.get(
            "https://itunes.apple.com/search",
            params={"term": f"{artist} {track}", "entity": "song", "limit": 5},
        )
        empty = {"artwork_path": None, "artwork_url": "", "track": "", "artist": "", "album": "", "year": ""}
        if r.status_code != 200:
            return empty
        results = (r.json() or {}).get("results") or []
        if not results:
            return empty
        artist_key = _tag_key(artist)
        track_key = _tag_key(track)
        canonical = next((res for res in results if _itunes_match(artist_key, track_key, res)), None)
        src = canonical or results[0]
        art = src.get("artworkUrl100") or ""
        art_hires = art.replace("100x100bb", "600x600bb").replace("100x100", "600x600") if art else ""
        out = dict(empty)
        if art_hires:
            out["artwork_url"] = art_hires  # URL for send_photo directly
            out["artwork_path"] = await self._download_image(art_hires, f"{video_id}_art.jpg")
        if canonical:  # gate passed — canonical fields confirmed, overwrite
            out["track"] = (canonical.get("trackName") or "").strip()
            out["artist"] = (canonical.get("artistName") or "").strip()
            out["album"] = (canonical.get("collectionName") or "").strip()
            out["year"] = _year_from_date(canonical.get("releaseDate"))
        return out

    async def _download_image(self, url: str, fname: str) -> str | None:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        path = TEMP_DIR / fname
        r = await self._http.get(url)
        if r.status_code != 200 or not r.content:
            return None
        path.write_bytes(r.content)
        return str(path)
