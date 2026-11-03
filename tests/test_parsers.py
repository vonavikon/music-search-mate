"""Unit tests for JSON parsing and _normalize."""
import pytest
from services.llm_search import _parse_json
from services.search import _normalize


class TestParseJSON:
    def test_clean_json(self):
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_json_with_markdown_wrapper(self):
        result = _parse_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_with_prefix_text(self):
        result = _parse_json('Some text before {"b": 2}')
        assert result == {"b": 2}

    def test_json_with_suffix_text(self):
        result = _parse_json('{"c": 3} trailing text')
        assert result == {"c": 3}

    def test_invalid_json_returns_none(self):
        assert _parse_json("not json at all") is None

    def test_empty_string(self):
        assert _parse_json("") is None

    def test_truncated_json(self):
        assert _parse_json('{"a": ') is None


class TestNormalize:
    def test_valid_entry(self):
        entry = {"id": "abc123", "title": "Test Song", "uploader": "Artist"}
        norm = _normalize(entry)
        assert norm["video_id"] == "abc123"
        assert norm["title"] == "Test Song"
        assert norm["channel"] == "Artist"

    def test_missing_id_returns_none(self):
        assert _normalize({"title": "No ID"}) is None

    def test_missing_title_returns_none(self):
        assert _normalize({"id": "abc"}) is None

    def test_duration_views_default_zero(self):
        entry = {"id": "abc", "title": "Song"}
        norm = _normalize(entry)
        assert norm["duration"] == 0
        assert norm["views"] == 0

    def test_channel_fallback(self):
        entry = {
            "id": "abc", "title": "S",
            "channel": "Channel Name",
            "uploader_name": "Uploader",
        }
        norm = _normalize(entry)
        assert norm["channel"] == "Channel Name"
