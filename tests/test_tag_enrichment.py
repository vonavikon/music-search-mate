"""Tests for tag_enrichment.enrich on m4a: no mp3-specific ID3 flags,
output goes to .tagged.m4a, tmp is cleaned up on any non-zero returncode."""
import subprocess
from pathlib import Path
from unittest.mock import patch

from services import tag_enrichment


def test_enrich_m4a_no_id3_flags(tmp_path):
    """enrich on m4a: no ID3 flags in the command, output written to .tagged.m4a, os.replace succeeded."""
    src = tmp_path / "abc123.m4a"
    src.write_bytes(b"fake m4a")

    def fake_run(args, capture_output=True, timeout=120):
        Path(args[-1]).write_bytes(b"tagged m4a")
        return subprocess.CompletedProcess(args, 0)

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        result = tag_enrichment.enrich(str(src), track="Test Song", artist="Test Artist")

    assert result == str(src)
    args = mock_run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert "-id3v2_version" not in args
    assert "-write_id3v2" not in args
    assert "-c:a" in args
    assert "copy" in args
    assert "-metadata" in args
    assert args[-1].endswith(".tagged.m4a")
    # os.replace: content moved back into the source, no tmp left behind
    assert src.read_bytes() == b"tagged m4a"
    assert not (tmp_path / "abc123.tagged.m4a").exists()


def test_enrich_m4a_cleanup_on_failure(tmp_path):
    """Non-zero returncode -> tmp cleaned up, source file untouched."""
    src = tmp_path / "abc123.m4a"
    src.write_bytes(b"original")

    def fake_run(args, capture_output=True, timeout=120):
        Path(args[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(args, 1)

    with patch("subprocess.run", side_effect=fake_run):
        result = tag_enrichment.enrich(str(src), track="Test Song", artist="Test Artist")

    assert result == str(src)
    assert src.read_bytes() == b"original"
    assert not (tmp_path / "abc123.tagged.m4a").exists()


def test_enrich_m4a_with_artwork_keeps_cover_atom(tmp_path):
    """With artwork: the covr atom (attached_pic) is kept, no ID3 flags."""
    src = tmp_path / "abc123.m4a"
    src.write_bytes(b"fake m4a")
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"fake jpg")

    def fake_run(args, capture_output=True, timeout=120):
        Path(args[-1]).write_bytes(b"tagged m4a")
        return subprocess.CompletedProcess(args, 0)

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        tag_enrichment.enrich(
            str(src), track="Test Song", artist="Test Artist", artwork=str(cover),
        )

    args = mock_run.call_args[0][0]
    assert "-disposition:v:0" in args
    assert "attached_pic" in args
    assert "-c:v" in args  # without an explicit encoder ffmpeg would pick h264, which the ipod muxer rejects
    assert "-id3v2_version" not in args
    assert "-write_id3v2" not in args
    assert args[-1].endswith(".tagged.m4a")
