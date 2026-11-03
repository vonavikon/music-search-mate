# Contributing

Issues and pull requests are welcome. The project is run by one person, so for anything larger than a small fix, open an issue first and describe what you want to change and why.

## Setup

Same as [Installing](README.md#installing): Python 3.12+ venv, keys in `.env` copied from `.env.example`.

## Tests

Tests run offline: external calls (LLM gateway, yt-dlp, Telegram, Last.fm) are mocked, no keys needed.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ --tb=short --ignore=tests/smoke_runner.py -q
```

CI runs the same command on every push and pull request; keep it green.

## Useful contributions

- Bug reports with the exact query text (or link) you sent and what you expected instead.
- "Wrong track" cases: your description, what the bot returned, and the track you meant — these feed the ranker calibration.
- Fixes with a test that reproduces the bug first.

## Discuss first

- Auto-ranking thresholds and daily limits: they are tuned on the economics of real traffic.
- Noise/moderation rules: every rule change trades false positives against misses.

## Commit style

`area: what changed`, lowercase, for example `handler: fix progress bar throttle`.
