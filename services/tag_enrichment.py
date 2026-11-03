"""Write canonical tags + artwork into the audio file.

Earlier the artist came from the YouTube channel, yielding
"hueylewisofficial". Enrichment is now a cascade (services/metadata.py):
yt-dlp info dict -> Last.fm -> iTunes artwork. Fields arriving here are
already canonical.

We overwrite: title (cleaned of noise), artist (canonical), album, date,
artwork. Audio is m4a (AAC native): tags are written as MP4 atoms
(-metadata), artwork via disposition attached_pic (covr atom). ID3-v2
flags (-id3v2_version/-write_id3v2) are mp3-specific and unused for MP4.
Artwork priority: album cover (iTunes) > video thumbnail (yt-dlp).
"""
import os
import re
import subprocess
from pathlib import Path

_NOISE_STRIP = re.compile(
    r"\s*[\(\|\[]\s*(official\s*(video|audio|music\s*video|lyric\s*video)|"
    r"music\s*video|hd|4k|hq|with\s*lyrics?|lyrics?|audio|video)\s*[\)\|\]]",
    re.I,
)
_PAREN_TRAILING = re.compile(r"\s*[\(\[\|]\s*[\)\]\|]\s*$")  # empty brackets left after strip


def _clean_title(t: str) -> str:
    t = _NOISE_STRIP.sub("", t)
    t = _PAREN_TRAILING.sub("", t)
    return t.strip(" -–—|")


def enrich(
    audio_path: str,
    *,
    track: str,
    artist: str,
    album: str = "",
    year: str = "",
    artwork: str | None = None,
    fallback_thumb: str | None = None,
) -> str:
    """Rewrite the audio file's tags in place (via tmp). On ffmpeg failure returns the original path.

    artwork - path to the album cover (iTunes); falls back to fallback_thumb
    (video thumbnail from yt-dlp). Input and output are the same container
    (m4a); the codec is unchanged.
    """
    p = Path(audio_path)
    if not p.exists():
        return audio_path

    clean_title = _clean_title(track) or track
    clean_artist = (artist or "").strip() or "Unknown Artist"

    cover = artwork if (artwork and Path(artwork).exists()) else fallback_thumb
    have_cover = bool(cover and Path(cover).exists())

    args = ["ffmpeg", "-y", "-i", audio_path]
    if have_cover:
        # -c:v copy is required: otherwise ffmpeg picks h264, which the m4a
        # ipod muxer rejects. For the covr atom the image is copied as-is
        # (jpg/mjpeg) + disposition attached_pic.
        args += ["-i", cover, "-map", "0:a", "-map", "1:0", "-c:a", "copy", "-c:v", "copy",
                 "-disposition:v:0", "attached_pic", "-metadata:s:v", "title=Album cover"]
    else:
        args += ["-c:a", "copy"]
    args += ["-metadata", f"title={clean_title}",
             "-metadata", f"artist={clean_artist}"]
    if album:
        args += ["-metadata", f"album={album}"]
    if year:
        args += ["-metadata", f"date={year}", "-metadata", f"year={year}"]
    tmp = str(p.with_suffix(".tagged.m4a"))
    args += [tmp]

    try:
        proc = subprocess.run(args, capture_output=True, timeout=120)
        if proc.returncode == 0 and Path(tmp).exists():
            os.replace(tmp, audio_path)
    except (subprocess.TimeoutExpired, OSError):
        pass
    finally:
        # Remove the tmp file on any outcome: success (already moved by
        # os.replace — no-op), non-zero returncode, or exception
        Path(tmp).unlink(missing_ok=True)
    return audio_path
