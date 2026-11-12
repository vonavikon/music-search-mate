"""Nov 2026 release announcement: texts, the bc: language toggle, sender dry-run."""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.broadcast_texts import BROADCAST_TEXTS, BROADCAST_MARK

REPO = Path(__file__).resolve().parent.parent


def test_texts_present_and_distinct():
    assert BROADCAST_TEXTS["ru"].strip()
    assert BROADCAST_TEXTS["en"].strip()
    assert BROADCAST_TEXTS["ru"] != BROADCAST_TEXTS["en"]


def test_texts_no_em_dash():
    # house style: no em dashes in user-facing text
    for text in BROADCAST_TEXTS.values():
        assert "—" not in text and "\u2014" not in text


@pytest.mark.asyncio
async def test_bc_callback_swaps_language():
    """bc:en edits the message to the EN text and offers an RU button back."""
    from bot.handler import handle_callback
    from telegram import User

    cq = MagicMock()
    cq.data = "bc:en"
    cq.answer = AsyncMock()
    cq.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = cq
    update.effective_user = User(id=123, is_bot=False, first_name="Test")
    context = MagicMock()
    context.bot_data = {"media": MagicMock()}
    context.chat_data = {}

    await handle_callback(update, context)

    cq.edit_message_text.assert_awaited_once()
    args, kwargs = cq.edit_message_text.await_args
    assert args[0] == BROADCAST_TEXTS["en"]
    (row,) = kwargs["reply_markup"].inline_keyboard
    assert row[0].text == "RU"
    assert row[0].callback_data == "bc:ru"


def _make_db(path):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE queries (user_id INTEGER, created_at REAL)")
    db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
               " created_at REAL NOT NULL, user_id INTEGER NOT NULL,"
               " query_id INTEGER, kind TEXT NOT NULL, video_id TEXT)")
    db.executemany("INSERT INTO queries(user_id) VALUES (?)",
                   [(1,), (2,), (3,), (2,)])
    db.commit()
    db.close()


def _dry(tmp_path):
    env = dict(os.environ, MSM_DB=str(tmp_path / "db.sqlite"),
               MSM_APP_ROOT=str(REPO))
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "broadcast.py"), "--dry"],
        env=env, capture_output=True, text=True, check=True)


# The sender script is private ops (not in the public mirror's include
# list); the dry-run test runs only where the file exists.
pytestmark = pytest.mark.skipif(
    not (REPO / "scripts" / "broadcast.py").exists(),
    reason="scripts/broadcast.py is not shipped in the public mirror")


def test_script_dry_counts_and_dedup(tmp_path):
    db_path = tmp_path / "db.sqlite"
    _make_db(db_path)
    r = _dry(tmp_path)
    assert "3 total, 0 already got it, 3 to go" in r.stdout

    db = sqlite3.connect(db_path)
    db.execute("INSERT INTO events(created_at, user_id, kind) VALUES (1, 2, ?)",
               (BROADCAST_MARK,))
    db.commit()
    db.close()
    r = _dry(tmp_path)
    assert "3 total, 1 already got it, 2 to go" in r.stdout
