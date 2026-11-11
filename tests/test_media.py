"""Integration tests for MediaService with the subprocess mocked out.

create_subprocess_exec is patched — these verify command construction,
exit-code handling, timeouts and stdout parsing.
"""
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from services.media import MediaService


@pytest.fixture
def media():
    return MediaService(max_concurrent=2)


@pytest.mark.asyncio
async def test_extract_url_meta_clean_url(media):
    """A clean YouTube URL extracts successfully."""
    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (
        json.dumps({
            "id": "abc123def45",
            "title": "Test Song",
            "uploader": "Test Artist",
            "duration": 300,
        }).encode(),
        b"",
    )

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await media.extract_url_meta("https://www.youtube.com/watch?v=abc123def45")

    assert result is not None
    assert result["video_id"] == "abc123def45"
    assert result["title"] == "Test Song"
    assert result["channel"] == "Test Artist"

    cmd = mock_exec.call_args[0]
    assert cmd[0].endswith("yt-dlp")  # venv binary resolved deterministically
    assert "--dump-json" in cmd


@pytest.mark.asyncio
async def test_extract_url_meta_shorts(media):
    """A Shorts URL: yt-dlp returns the id and nothing breaks."""
    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (
        json.dumps({"id": "short123", "title": "Short", "uploader": "U", "duration": 30}).encode(),
        b"",
    )

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await media.extract_url_meta("https://www.youtube.com/shorts/short123")

    assert result is not None
    assert result["video_id"] == "short123"


@pytest.mark.asyncio
async def test_extract_url_meta_failure(media):
    """yt-dlp returns a non-zero exit code -> None."""
    proc = AsyncMock()
    proc.returncode = 1
    proc.communicate.return_value = (b"", b"ERROR: video not found")

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await media.extract_url_meta("https://www.youtube.com/watch?v=bad")

    assert result is None


@pytest.mark.asyncio
async def test_extract_url_meta_timeout(media):
    """yt-dlp hangs -> timeout -> None."""
    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (b"{}", b"")
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            result = await media.extract_url_meta("https://www.youtube.com/watch?v=slow")

    assert result is None
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_download_audio_success(media, tmp_path, monkeypatch):
    """download_audio builds the correct command and finds the .m4a in TEMP_DIR."""
    monkeypatch.setattr("config.TEMP_DIR", tmp_path)
    monkeypatch.setattr("services.media.TEMP_DIR", tmp_path)

    video_id = "test123"
    url = f"https://www.youtube.com/watch?v={video_id}"
    fake_m4a = tmp_path / f"{video_id}.m4a"
    fake_m4a.write_text("fake m4a data")
    fake_info = tmp_path / f"{video_id}.info.json"
    fake_info.write_text(json.dumps({"id": video_id, "title": "Test", "uploader": "Artist"}))

    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (b"", b"")

    # yt-dlp creates the file named by the id from the URL — the exact path is unknown, so it is found by an iterdir scan
    with patch("pathlib.Path.iterdir", return_value=[fake_m4a, fake_info]):
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            m4a_path, thumb_path, info = await media.download_audio(url)

    cmd = mock_exec.call_args[0]
    assert cmd[0].endswith("yt-dlp")  # venv binary resolved deterministically
    assert "--extract-audio" in cmd
    assert "--audio-format" in cmd
    assert "m4a" in cmd
    # maximum quality: prefer native m4a + VBR-0, with an mp3 320K fallback pass
    assert cmd[cmd.index("--audio-quality") + 1] == "0"
    assert cmd[cmd.index("-f") + 1] == "bestaudio"  # top format, not locked to m4a
    assert url in str(cmd)

    assert m4a_path.endswith(".m4a")
    assert "id" in info
    assert info.get("title") == "Test"        # regression: .info.json used to be matched by f.suffix
    assert info.get("uploader") == "Artist"   # (suffix ".json" instead of ".info.json")
    assert thumb_path is None  # no .jpg in the scan


@pytest.mark.asyncio
async def test_download_audio_403(media, tmp_path, monkeypatch):
    """Download returns a non-zero exit code -> RuntimeError."""
    monkeypatch.setattr("config.TEMP_DIR", tmp_path)
    monkeypatch.setattr("services.media.TEMP_DIR", tmp_path)

    proc = AsyncMock()
    proc.returncode = 1
    proc.communicate.return_value = (b"", b"HTTP Error 403: Forbidden")

    with patch("pathlib.Path.iterdir", return_value=[]):
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="403"):
                await media.download_audio("https://www.youtube.com/watch?v=forbidden")


@pytest.mark.asyncio
async def test_download_audio_soundcloud_url(media, tmp_path, monkeypatch):
    """A SoundCloud URL: yt-dlp handles the platform on its own."""
    monkeypatch.setattr("config.TEMP_DIR", tmp_path)
    monkeypatch.setattr("services.media.TEMP_DIR", tmp_path)

    sc_id = "sc_track_123"
    url = "https://soundcloud.com/someartist/sometrack"
    fake_m4a = tmp_path / f"{sc_id}.m4a"
    fake_m4a.write_text("fake")
    fake_info = tmp_path / f"{sc_id}.info.json"
    fake_info.write_text(json.dumps({"id": sc_id, "title": "SC Track"}))

    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (b"", b"")

    with patch("pathlib.Path.iterdir", return_value=[fake_m4a, fake_info]):
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            m4a_path, _, info = await media.download_audio(url)

    assert m4a_path.endswith(".m4a")
    assert "id" in info
    assert info.get("title") == "SC Track"    # regression: .info.json lookup by name
    cmd = mock_exec.call_args[0]
    assert url in str(cmd)


@pytest.mark.asyncio
async def test_download_audio_timeout(media, tmp_path, monkeypatch):
    """Download hangs -> timeout -> RuntimeError."""
    monkeypatch.setattr("config.TEMP_DIR", tmp_path)
    monkeypatch.setattr("services.media.TEMP_DIR", tmp_path)

    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (b"", b"")
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(RuntimeError, match="timeout"):
                await media.download_audio("https://www.youtube.com/watch?v=slow")

    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_download_audio_uses_dedicated_subdir(media, tmp_path, monkeypatch):
    """yt-dlp writes into a unique dl_* subdirectory under TEMP_DIR, not the shared directory."""
    monkeypatch.setattr("config.TEMP_DIR", tmp_path)
    monkeypatch.setattr("services.media.TEMP_DIR", tmp_path)

    video_id = "subdir123"
    url = f"https://www.youtube.com/watch?v={video_id}"
    fake_m4a = tmp_path / f"{video_id}.m4a"
    fake_m4a.write_text("fake")

    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (b"", b"")

    with patch("pathlib.Path.iterdir", return_value=[fake_m4a]):
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            m4a_path, _, _ = await media.download_audio(url)

    cmd = mock_exec.call_args[0]
    outtmpl = cmd[cmd.index("-o") + 1]
    parent = Path(outtmpl).parent
    assert parent.is_relative_to(tmp_path)  # inside TEMP_DIR
    assert parent.name.startswith("dl_")     # unique per-download subdirectory
    assert m4a_path.endswith(".m4a")


def test_cleanup_temp_recursive(monkeypatch, tmp_path):
    """Files INSIDE dl_* directories older than the TTL are deleted (previously they never were)."""
    import os
    import time as _t
    from services import media
    monkeypatch.setattr(media, "TEMP_DIR", tmp_path)
    old = _t.time() - 10_000
    nested = tmp_path / "dl_abc"
    nested.mkdir()
    f1 = nested / "track.m4a"
    f1.write_bytes(b"x")
    os.utime(f1, (old, old))
    fresh = tmp_path / "dl_inprogress"
    fresh.mkdir()
    f2 = fresh / "track2.m4a"
    f2.write_bytes(b"y")
    n = media.cleanup_temp(ttl_minutes=1)
    assert not f1.exists() and not nested.exists()   # stale content removed along with its directory
    assert f2.exists()                                # fresh download untouched
    assert n >= 2


# ════ F6: video quality selection ════

def test_video_format_per_quality():
    """The -f selector is built per quality; 'best' keeps today's cascade."""
    from services.media import video_format
    assert video_format("240") == (
        "bv*[height<=240][ext=mp4]+ba[ext=m4a]"
        "/bv*[height<=240]+ba"
        "/b[height<=240]")
    assert video_format("360") == (
        "bv*[height<=360][ext=mp4]+ba[ext=m4a]"
        "/bv*[height<=360]+ba"
        "/b[height<=360]")
    assert video_format("480") == (
        "bv*[height<=480][ext=mp4]+ba[ext=m4a]"
        "/bv*[height<=480]+ba"
        "/b[height<=480]")
    assert video_format("360p") == video_format("360")  # old dl:q style tolerated
    assert video_format("best") == (
        "best[height<=720][ext=mp4]"
        "/bestvideo[height<=720]+bestaudio"
        "/best[height<=720]"
        "/best[ext=mp4]/best")


def test_video_format_legacy_heights_keep_cascade():
    """720/1080/1440 (old dl:q buttons) keep the height-capped legacy cascade."""
    from services.media import video_format
    assert video_format("1080") == (
        "best[height<=1080][ext=mp4]"
        "/bestvideo[height<=1080]+bestaudio"
        "/best[height<=1080]"
        "/best[ext=mp4]/best")
    assert video_format("720") == video_format("best")


@pytest.mark.asyncio
async def test_download_video_default_quality_and_args(media, tmp_path, monkeypatch):
    """Default call: 360p selector, mp4 merge, socket timeout, 50M size cap,
    and the chosen label comes back in info."""
    from services.media import video_format
    monkeypatch.setattr("services.media.TEMP_DIR", tmp_path)
    monkeypatch.delenv("VIDEO_MAX_FILESIZE", raising=False)

    vid = "dQw4w9WgXcQ"
    url = f"https://www.youtube.com/watch?v={vid}"
    fake_mp4 = tmp_path / f"{vid}.mp4"
    fake_mp4.write_bytes(b"v")
    fake_info = tmp_path / f"{vid}.info.json"
    fake_info.write_text(json.dumps({"id": vid, "title": "Test"}))

    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (b"", b"")

    with patch("pathlib.Path.iterdir", return_value=[fake_mp4, fake_info]):
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            path, info = await media.download_video(url)

    cmd = mock_exec.call_args[0]
    # the media-layer signature default is 360 (the handler always passes the
    # height picked in the quality picker)
    assert cmd[cmd.index("-f") + 1] == video_format("360")
    assert cmd[cmd.index("--merge-output-format") + 1] == "mp4"
    assert "--socket-timeout" in cmd
    assert cmd[cmd.index("--max-filesize") + 1] == "50M"
    assert path.endswith(".mp4")
    assert info["chosen_quality"] == "360"


@pytest.mark.asyncio
async def test_download_video_oversize_steps_down(media, tmp_path, monkeypatch):
    """A merged file over ~48MB at 480p retries one step lower (360p); the
    oversized directory is dropped immediately."""
    from services.media import video_format
    monkeypatch.setattr("services.media.TEMP_DIR", tmp_path)
    monkeypatch.delenv("VIDEO_MAX_FILESIZE", raising=False)

    vid = "bigvideo0001"
    url = f"https://www.youtube.com/watch?v={vid}"
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()
    big = run1 / "v.mp4"
    with open(big, "wb") as f:
        f.truncate(49 * 1024 * 1024)  # sparse: st_size 49MB, no real disk cost
    small = run2 / "v.mp4"
    small.write_bytes(b"v")
    info1 = run1 / "v.info.json"
    info1.write_text(json.dumps({"id": vid}))
    info2 = run2 / "v.info.json"
    info2.write_text(json.dumps({"id": vid}))

    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (b"", b"")

    with patch("pathlib.Path.iterdir", side_effect=[[big, info1], [small, info2]]):
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            path, info = await media.download_video(url, "480")

    assert Path(path).name == "v.mp4"
    assert path == str(small)
    assert info["chosen_quality"] == "360"
    fmts = []
    for c in mock_exec.call_args_list:
        cmd = c[0]
        fmts.append(cmd[cmd.index("-f") + 1])
    assert fmts == [video_format("480"), video_format("360")]
    assert not run1.exists()  # the oversized dir doesn't wait for TTL cleanup


@pytest.mark.asyncio
async def test_download_video_oversize_at_240_raises(media, tmp_path, monkeypatch):
    """Chain bottom reached while still over the guard: refuse instead of
    delivering a file Telegram would reject."""
    monkeypatch.setattr("services.media.TEMP_DIR", tmp_path)
    monkeypatch.delenv("VIDEO_MAX_FILESIZE", raising=False)

    vid = "hugevideo001"
    url = f"https://www.youtube.com/watch?v={vid}"
    run1 = tmp_path / "run1"
    run1.mkdir()
    big = run1 / "v.mp4"
    with open(big, "wb") as f:
        f.truncate(49 * 1024 * 1024)
    info1 = run1 / "v.info.json"
    info1.write_text(json.dumps({"id": vid}))

    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (b"", b"")

    with patch("pathlib.Path.iterdir", side_effect=[[big, info1]]):
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            with pytest.raises(RuntimeError, match="even at 240p"):
                await media.download_video(url, "240")

    assert mock_exec.await_count == 1  # nothing below 240 in the chain
    assert not run1.exists()
