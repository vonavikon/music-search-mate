"""Audio, video and thumbnail downloads via a yt-dlp subprocess.

Data flow steps 6-7. A semaphore caps concurrent downloads; a TTL cleanup
keeps the temp directory from filling the disk (a lesson from an earlier
project where temp files did exactly that). URL fast path: extract_url_meta
plus download_audio(url) handles links without touching the LLM.
"""
import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from config import TEMP_DIR
from services.observability import CircuitBreaker

_log = logging.getLogger("bot.media")


# yt-dlp counters: share of successful downloads in production. Process-local
# (single-process bot), reset on restart. Error % is visible from the ok/fail pair in logs.
_YTDLP_OK = 0
_YTDLP_FAIL = 0


def ytdlp_stats() -> dict:
    """Download counters: {ok, fail, error_rate}. For logs/healthcheck."""
    total = _YTDLP_OK + _YTDLP_FAIL
    return {"ok": _YTDLP_OK, "fail": _YTDLP_FAIL,
            "error_rate": (_YTDLP_FAIL / total) if total else 0.0}


async def _run_ytdlp(args, timeout: int, on_progress=None):
    """Run yt-dlp with live progress. on_progress(pct) fires no more often than
    once per 2s and no less than once every 5%. Without a callback — the previous
    quiet path (test compatibility). Returns (returncode, stderr_tail)."""
    if on_progress is None:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        return proc.returncode, stderr

    import re as _re
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    err_buf: list = []

    async def _drain_err():
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                break
            err_buf.append(chunk)
            if sum(len(c) for c in err_buf) > 100_000:
                del err_buf[:1]

    async def _drain_out():
        last_t, last_p = 0.0, -5.0
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            m = _re.search(rb"\[download\]\s+(\d{1,3}(?:\.\d+)?)%",
                           line)
            if not m:
                continue
            pct = float(m.group(1))
            now = time.time()
            if (pct >= last_p + 3 and now - last_t >= 1.2) or pct >= 100:
                last_p, last_t = pct, now
                try:
                    await on_progress(pct)
                except Exception:
                    pass

    t_err = asyncio.create_task(_drain_err())
    t_out = asyncio.create_task(_drain_out())
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    finally:
        t_err.cancel()
        t_out.cancel()
        await asyncio.gather(t_err, t_out, return_exceptions=True)
    return proc.returncode, b"".join(err_buf)[-400:]


class MediaService:
    def __init__(self, max_concurrent: int):
        self._sem = asyncio.Semaphore(max_concurrent)
        # Video gets a dedicated slot: video downloads take minutes and must not
        # stall the audio queue (audio ~20s per slot). Plus a busy-guard in the handler.
        self._vsem = asyncio.Semaphore(1)
        self._breaker = CircuitBreaker("yt-dlp", threshold=3, cooldown=60.0)

    async def download_video(self, url: str, quality: str = "720p",
                             on_progress=None) -> tuple[str, dict]:
        """Download a VIDEO by URL at the given quality.
        Returns (mp4_path, info_dict). Codec cascade avc1→vp9→av01."""
        global _YTDLP_OK, _YTDLP_FAIL
        if self._breaker.is_open:
            raise RuntimeError("yt-dlp circuit open — download skipped")
        dl_dir = Path(tempfile.mkdtemp(dir=TEMP_DIR, prefix="dl_v_"))
        outtmpl = str(dl_dir / "%(id)s.%(ext)s")

        height_map = {"240p": 240, "360p": 360, "480p": 480,
                      "720p": 720, "1080p": 1080, "1440p": 1440}
        h = height_map.get(quality, 720)
        # format: video+audio in mp4 (native merge if available, otherwise an ffmpeg merge)
        fmt = (f"best[height<={h}][ext=mp4]"
               f"/bestvideo[height<={h}]+bestaudio"
               f"/best[height<={h}]"
               f"/best[ext=mp4]/best")

        # With a local Bot API server the limit is 2 GB, without it 50 MB.
        # Env, not cfg: media.py doesn't know the bot config (single source — .env).
        max_fs = os.getenv("VIDEO_MAX_FILESIZE", "50M")
        args = [
            "yt-dlp", "--socket-timeout", "30",
            "-f", fmt,
            "--no-playlist", "--no-warnings", "--no-progress",
            "--write-info-json", "--merge-output-format", "mp4",
            "--max-filesize", max_fs,
            "-o", outtmpl, url,
        ]
        cookies = os.getenv('YTDLP_COOKIEFILE')
        if cookies and Path(cookies).exists():
            args[1:1] = ['--cookies', cookies]

        # 2 GB at a slow transfer speed takes > 300s: the timeout must cover the limit's volume.
        # Live progress via _run_ytdlp ([download] X% into the status).
        # --no-progress silences [download] entirely (live run in the container:
        # 0 events) — with a callback switch to --progress, without a callback
        # keep the previous quiet mode.
        if on_progress:
            args = [a for a in args if a != "--no-progress"] + ["--progress"]
        args[1:1] = ["--newline"]
        async with self._vsem:
            try:
                rc, stderr = await _run_ytdlp(args, 600, on_progress)
            except asyncio.TimeoutError:
                _YTDLP_FAIL += 1
                self._breaker.record_failure()
                shutil.rmtree(dl_dir, ignore_errors=True)  # GB-scale garbage doesn't wait for TTL
                raise RuntimeError(f"yt-dlp video timeout for {url}")

        if rc != 0:
            _YTDLP_FAIL += 1
            self._breaker.record_failure()
            err = stderr.decode(errors="ignore")[-200:] if stderr else "?"
            shutil.rmtree(dl_dir, ignore_errors=True)  # garbage doesn't wait for TTL
            raise RuntimeError(f"yt-dlp video failed: {err}")

        _YTDLP_OK += 1
        self._breaker.record_success()

        mp4_path = None
        info = {}
        for f in dl_dir.iterdir():
            if f.name.endswith(".info.json"):
                try:
                    info = json.loads(f.read_text())  # a corrupt json must not kill the download
                except (OSError, json.JSONDecodeError):
                    info = {}
                # video info.json used to live until TTL (the audio path deletes
                # its own) — clean up right after reading
                try:
                    f.unlink()
                except OSError:
                    pass
            elif f.suffix == ".mp4":
                mp4_path = str(f)
        if not mp4_path:
            shutil.rmtree(dl_dir, ignore_errors=True)  # a .part file without an mp4
            raise RuntimeError("yt-dlp video: no mp4 output")
        _log.info("yt-dlp video ok: %s (%d bytes)", Path(mp4_path).name,
                  Path(mp4_path).stat().st_size)
        return mp4_path, info

    async def download_audio(self, url: str,
                             on_progress=None) -> tuple[str, str | None, dict]:
        """Returns (m4a_path, thumb_path | None, info_dict).
        url is the full video/track URL (YouTube, SoundCloud, etc.).
        info_dict comes from {id}.info.json (yt-dlp --write-info-json): artist/track/
        album/release_date/thumbnails. Note: --dump-json won't do - it implies
        simulate and no audio is created. --write-info-json writes the file on a real download."""
        global _YTDLP_OK, _YTDLP_FAIL
        # circuit breaker: when yt-dlp is dead (rate-limit/403 wave) fail fast
        # instead of waiting out the 180s timeout on each of 3+ candidates in a row.
        if self._breaker.is_open:
            raise RuntimeError("yt-dlp circuit open — download skipped (degraded)")
        # unique subfolder per download: with max_concurrent=2 two parallel downloads
        # from different chats don't pick up each other's files (TEMP_DIR.iterdir race)
        dl_dir = Path(tempfile.mkdtemp(dir=TEMP_DIR, prefix="dl_"))
        outtmpl = str(dl_dir / "%(id)s.%(ext)s")
        # Owner's request: "the highest available quality" — NO container preference.
        # The old m4a fix grabbed AAC 128k (itag 140) even though YouTube also serves
        # higher-quality Opus ~160k (251). yt-dlp picks the top audio format itself
        # (usually Opus 160k; SoundCloud 256k+; AAC 256k with Premium cookies); we convert
        # to m4a at max VBR (quality 0); if the best is already m4a, remux without re-encoding.
        # Fallback on error: mp3 320K.
        audio_args = [
            "-f", "bestaudio",
            "--extract-audio", "--audio-format", "m4a", "--audio-quality", "0",
        ]
        args = [
            "yt-dlp", "--socket-timeout", "30",
            *audio_args,
            "--no-playlist", "--no-warnings",
            "--no-progress", "--write-thumbnail", "--convert-thumbnails", "jpg",
            "--write-info-json",
            "-o", outtmpl, url,
        ]
        # Logged-in session cookies pass YouTube's datacenter-IP bot-check
        # (LOGIN_REQUIRED wave since 2026-10-06). The file is mounted from ./secrets;
        # no file -> the flag is not added, previous behavior.
        cookies = os.getenv('YTDLP_COOKIEFILE')
        if cookies and Path(cookies).exists():
            args[1:1] = ['--cookies', cookies]

        # Same fix as for video: swap --no-progress for --progress when a live
        # callback is set (otherwise [download] X% is not printed at all).
        if on_progress:
            args = [a for a in args if a != "--no-progress"] + ["--progress"]
        args[1:1] = ["--newline"]  # line-by-line [download] X%
        async with self._sem:
            try:
                rc, stderr = await _run_ytdlp(args, 180, on_progress)
            except asyncio.TimeoutError:
                _YTDLP_FAIL += 1
                self._breaker.record_failure()
                shutil.rmtree(dl_dir, ignore_errors=True)  # garbage doesn't wait for TTL
                raise RuntimeError(f"yt-dlp timeout for {url}")

        # With --audio-format m4a yt-dlp creates a .m4a file named after the id in the URL
        # (for YouTube: https://www.youtube.com/watch?v=VIDEO_ID -> id=VIDEO_ID;
        #  for SoundCloud the id comes from the yt-dlp extractor). The exact id is unknown
        #  until the run — scan dl_dir for finished files instead of constructing the path.
        #  dl_dir is not removed: _send_track reads the files after return; TTL cleanup is cleanup_temp's job.
        info_json_path = None
        m4a_path = None
        thumb_path = None

        def _scan():
            nonlocal info_json_path, m4a_path, thumb_path
            for f in dl_dir.iterdir():
                # f.suffix of "x.info.json" is ".json", so match on the full name
                if f.name.endswith(".info.json"):
                    info_json_path = f
                elif f.suffix in (".m4a", ".mp3"):
                    m4a_path = str(f)
                elif f.suffix == ".jpg":
                    thumb_path = str(f)

        _scan()
        if m4a_path is None:
            # fallback run: mp3 320K. The premature _YTDLP_FAIL increment was
            # removed — the final increment below counts it once per incident
            retry = [a for a in args]
            i = retry.index("--audio-format")
            retry[i + 1] = "mp3"
            j = retry.index("--audio-quality")
            retry[j + 1] = "320K"
            # the retry used to run without a timeout and outside the semaphore —
            # a hung yt-dlp stalled _send_track forever (the "54 min" incident)
            async with self._sem:
                try:
                    rc2, _ = await _run_ytdlp(retry, 180)
                except asyncio.TimeoutError:
                    _YTDLP_FAIL += 1
                    self._breaker.record_failure()
                    shutil.rmtree(dl_dir, ignore_errors=True)  # garbage doesn't wait for TTL
                    raise RuntimeError(f"yt-dlp mp3-fallback timeout for {url}")
            rc = rc or rc2
            _scan()

        if rc != 0 or not m4a_path:
            tail = (stderr or b"").decode(errors="ignore")[-400:]
            _YTDLP_FAIL += 1
            self._breaker.record_failure()
            shutil.rmtree(dl_dir, ignore_errors=True)  # garbage doesn't wait for TTL
            raise RuntimeError(f"yt-dlp failed for {url}: {tail}")

        info: dict = {}
        if info_json_path:
            try:
                info = json.loads(Path(info_json_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                info = {}
            try:
                Path(info_json_path).unlink(missing_ok=True)
            except OSError:
                pass
        if not info:
            info = {"id": url.rsplit("=", 1)[-1] if "=" in url else url}
        _YTDLP_OK += 1
        self._breaker.record_success()
        # temp size is a metric: TTL-cleanup keeps temp from growing unbounded.
        try:
            size_kb = Path(m4a_path).stat().st_size / 1024
            _log.info("yt-dlp ok: %s (%.0f KB, ok=%d fail=%d)",
                      Path(m4a_path).name, size_kb, _YTDLP_OK, _YTDLP_FAIL)
        except OSError:
            pass
        return m4a_path, (thumb_path if thumb_path and Path(thumb_path).exists() else None), info

    async def extract_url_meta(self, url: str) -> dict | None:
        """Video metadata by direct URL (no download). For the URL fast-path.

        Returns {video_id, title, channel, duration} or None on failure.
        The existing download_audio then works from video_id — minimal new code.
        """
        args = ["yt-dlp", "--socket-timeout", "30", "--dump-json", "--no-playlist", "--no-warnings", url]
        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=90)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return None
        if proc.returncode != 0 or not out:
            return None
        first = out.decode(errors="ignore").strip().splitlines()[0]
        try:
            info = json.loads(first)
        except json.JSONDecodeError:
            return None
        vid = info.get("id") or ""
        if not vid:
            return None
        return {
            "video_id": vid,
            "title": (info.get("title") or "").strip(),
            "channel": (info.get("uploader") or info.get("channel") or "").strip(),
            "duration": info.get("duration") or 0,
            "extractor": (info.get("extractor") or "").strip(),
        }


def cleanup_temp(ttl_minutes: int = 0) -> int:
    """Delete files and empty dl_* subfolders in TEMP older than ttl_minutes.
    Returns the number of removed items."""
    cutoff = time.time() - ttl_minutes * 60
    n = 0
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    # Recursive: files inside dl_* older than TTL were never deleted (rmdir only
    # took empty dirs), post-crash garbage piled up to the tmpfs limit.
    # Fresh files (download in progress) are untouched: the cut is by mtime.
    for root, dirs, files in os.walk(TEMP_DIR, topdown=False):
        for name in files:
            f = os.path.join(root, name)
            try:
                if os.path.getmtime(f) < cutoff:
                    os.unlink(f)
                    n += 1
            except OSError:
                pass
        # an empty dl_* goes without an mtime gate: deleting the last file updates
        # the dir's mtime itself, so an "old empty" dir would never be removed
        if root != str(TEMP_DIR) and os.path.basename(root).startswith("dl_"):
            try:
                if not os.listdir(root):
                    os.rmdir(root)
                    n += 1
            except OSError:
                pass
    # size-guard: if the TTL mechanism ever breaks (past incident: temp filled
    # the disk), hard-delete everything older than 60 seconds once the threshold is exceeded.
    try:
        _size = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(TEMP_DIR) for f in fs)
        if _size > 10 * 1024 ** 3:
            _hard = time.time() - 60
            for root, dirs, files in os.walk(TEMP_DIR, topdown=False):
                for name in files:
                    f = os.path.join(root, name)
                    try:
                        if os.path.getmtime(f) < _hard:
                            os.unlink(f)
                            n += 1
                    except OSError:
                        pass
            _log.warning("temp size-guard: %.1f GB, removed %d files",
                         _size / 1024 ** 3, n)
    except OSError:
        pass
    return n
