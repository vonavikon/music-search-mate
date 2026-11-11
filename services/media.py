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
import re
import shutil
import sys
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

# Deterministic binary: the interpreter's own environment (venv) ships the
# pinned yt-dlp. A stale yt-dlp earlier in PATH (homebrew 2026.07 in dev)
# silently broke the EJS challenge solver -> storyboards-only formats.
_YTDLP_BIN = str(Path(sys.executable).parent / "yt-dlp")
if not Path(_YTDLP_BIN).is_file():
    _YTDLP_BIN = "yt-dlp"


# Video quality: the handler's quality picker always passes the user's explicit
# height (240-1440); this default only backs a malformed callback payload.
# Belongs in config.py as VIDEO_DEFAULT_QUALITY (default "360") once config.py
# is in scope for this change — module constant for now.
_DEFAULT_VIDEO_QUALITY = "360"

# Heights with the compact height-capped selectors (the picker's 240-480
# buttons). Higher picks ("best", 720/1080/1440) keep the legacy cascade.
_VIDEO_COMPACT_MAX_HEIGHT = 480

# Step-down chain when a merged file still exceeds the size guard (~96% of the
# --max-filesize limit, 48MB of the 50MB Bot API cap): re-fetching one step
# lower is cheaper than an upload Telegram will reject.
_VIDEO_FALLBACK_CHAIN = ("best", "480", "360", "240")


def _norm_video_quality(quality: str) -> str:
    """'360p'/'360' -> '360'; 'best'/''/anything non-numeric -> 'best'."""
    q = (quality or "").strip().lower()
    if q.endswith("p") and q[:-1].isdigit():
        q = q[:-1]
    return q if q.isdigit() else "best"


def video_format(quality: str) -> str:
    """yt-dlp -f selector for a quality label (unit-tested per quality).

    240-480 (the picker's compact buttons): height-capped selectors — separate
    mp4+m4a with a native merge, then any containers via an ffmpeg merge, then
    a progressive stream. "best" and higher explicit heights (720/1080/1440):
    today's codec cascade unchanged."""
    q = _norm_video_quality(quality)
    if q.isdigit() and int(q) <= _VIDEO_COMPACT_MAX_HEIGHT:
        h = int(q)
        return (f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]"
                f"/bv*[height<={h}]+ba"
                f"/b[height<={h}]")
    h = int(q) if q.isdigit() else 720
    return (f"best[height<={h}][ext=mp4]"
            f"/bestvideo[height<={h}]+bestaudio"
            f"/best[height<={h}]"
            f"/best[ext=mp4]/best")


def _filesize_bytes(spec: str) -> int:
    """'50M' -> bytes ('2G' with a local Bot API server). Unparsable -> 50MB."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*$", (spec or "").upper())
    if not m:
        return 50 * 1024 ** 2
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}[m.group(2)]
    return int(float(m.group(1)) * mult)


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

    @staticmethod
    def _ytdlp_net_args() -> list[str]:
        """Shared network flags for every yt-dlp call.

        EJS challenge solver: since 2026-11 YouTube returns storyboards-only
        ("Requested format is not available") unless the n-challenge is solved;
        the solver script is fetched from GitHub once and cached by yt-dlp.
        Cookies: logged-in session passes the datacenter-IP bot-check;
        without the file the flag is not added (previous behavior)."""
        args = ["--remote-components", "ejs:github"]
        cookies = os.getenv('YTDLP_COOKIEFILE')
        if cookies and Path(cookies).exists():
            args += ['--cookies', cookies]
        return args

    async def download_video(self, url: str, quality: str = _DEFAULT_VIDEO_QUALITY,
                             on_progress=None) -> tuple[str, dict]:
        """Download a VIDEO by URL at the given quality.

        quality: the height picked in the handler's picker ("240"-"1440") or
        "best"; a trailing "p" from old dl:q buttons is tolerated. Size guard:
        when the merged file exceeds ~96% of the --max-filesize limit (48MB of
        the 50MB Bot API cap), the download steps one quality lower and
        retries — unlikely at 240/360, the chain is best -> 480 -> 360 -> 240.
        Returns (mp4_path, info_dict) where info gains "chosen_quality", the
        label that actually produced the file."""
        global _YTDLP_OK
        if self._breaker.is_open:
            raise RuntimeError("yt-dlp circuit open — download skipped")
        q = _norm_video_quality(quality)
        if q in _VIDEO_FALLBACK_CHAIN:
            chain = _VIDEO_FALLBACK_CHAIN[_VIDEO_FALLBACK_CHAIN.index(q):]
        else:
            # an explicit height above the chain (720/1080/1440 from old buttons):
            # try it first, then step into the chain; 720 IS today's best selector,
            # so don't run the same selector twice
            tail = _VIDEO_FALLBACK_CHAIN
            if video_format(q) == video_format("best"):
                tail = tail[1:]
            chain = (q,) + tail
        # With a local Bot API server the limit is 2 GB, without it 50 MB.
        # Env, not cfg: media.py doesn't know the bot config (single source — .env).
        max_fs = os.getenv("VIDEO_MAX_FILESIZE", "50M")
        guard = int(_filesize_bytes(max_fs) * 0.96)  # ~48MB of the 50MB limit
        path, info = "", {}
        for step, ql in enumerate(chain):
            path, info = await self._download_video_once(url, ql, max_fs, on_progress)
            try:
                size = Path(path).stat().st_size
            except OSError:
                size = 0
            if size <= guard:
                break
            # over the guard even after the merge: drop the file and retry one
            # step lower (the Bot API would reject the upload anyway)
            shutil.rmtree(Path(path).parent, ignore_errors=True)
            if step == len(chain) - 1:
                raise RuntimeError(f"yt-dlp video: over {max_fs} even at {ql}p "
                                   f"({size / 1_048_576:.0f} MB)")
            _log.info("video %sp merged to %.0f MB (guard %.0f MB) — retrying one step lower",
                      ql, size / 1_048_576, guard / 1_048_576)
        _YTDLP_OK += 1
        self._breaker.record_success()
        info["chosen_quality"] = chain[step]
        return path, info

    async def _download_video_once(self, url: str, quality: str, max_fs: str,
                                   on_progress=None) -> tuple[str, dict]:
        """One yt-dlp video run at a fixed quality token. Raises RuntimeError on
        any yt-dlp failure (FAIL counter + breaker, same as before); the
        size-guard step-down lives in download_video."""
        global _YTDLP_FAIL
        dl_dir = Path(tempfile.mkdtemp(dir=TEMP_DIR, prefix="dl_v_"))
        outtmpl = str(dl_dir / "%(id)s.%(ext)s")

        # format: video+audio in mp4 (native merge if available, otherwise an ffmpeg merge)
        fmt = video_format(quality)
        args = [
            _YTDLP_BIN, "--socket-timeout", "30",
            "-f", fmt,
            "--no-playlist", "--no-warnings", "--no-progress",
            "--write-info-json", "--merge-output-format", "mp4",
            "--max-filesize", max_fs,
            "-o", outtmpl, url,
        ]
        args[1:1] = self._ytdlp_net_args()

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
            _YTDLP_BIN, "--socket-timeout", "30",
            *audio_args,
            "--no-playlist", "--no-warnings",
            "--no-progress", "--write-thumbnail", "--convert-thumbnails", "jpg",
            "--write-info-json",
            "-o", outtmpl, url,
        ]
        # Cookies + EJS solver via the shared helper (see _ytdlp_net_args).
        args[1:1] = self._ytdlp_net_args()

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
        args = [_YTDLP_BIN, "--socket-timeout", "30", "--dump-json", "--no-playlist",
                "--no-warnings", *self._ytdlp_net_args(), url]
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
