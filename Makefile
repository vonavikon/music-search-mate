# music-search-bot test pipeline
# Gate 1: fast mocked pytest (<5 sec, no network)
# Gate 2: real bothub + yt-dlp smoke tests (~120 sec)

.PHONY: test smoke smoke-fast check

test:
	python -m pytest tests/ -v --tb=short --timeout=30 --ignore=tests/smoke_runner.py

smoke:
	@PYTHONUTF8=1 python tests/smoke_runner.py

smoke-fast:
	@PYTHONUTF8=1 python tests/smoke_runner.py --no-judge

check: test smoke-fast
	@echo "=== Full gate complete ==="
