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
    assert "yt-dlp" in cmd
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
    assert "yt-dlp" in cmd
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
