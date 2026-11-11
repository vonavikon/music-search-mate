"""Observability for Phase 2: circuit breaker + PII redaction.

Circuit breaker — fast-fail on a dead upstream (bothub/yt-dlp): after N consecutive
failures the circuit opens, network calls are skipped and a degraded response is
returned instead of 3 retries × timeout per request. After the cooldown — half-open
(one probe).

PII redaction — logging policy: the user's description (free text) may contain
email/phone/names. Logs get only a truncated, scrubbed copy; the full description
lives in queries.description (data, not logs) and is needed for conversation context.
"""
import logging
import re
import time

_log = logging.getLogger("bot.obs")


# ---------- neutral tech-metric hooks (wave 45-3) ----------
# services/ does NOT depend on web/: the engine only EMITS events (search
# stage, external call, gauge delta), and the subscriber, web/metrics.py,
# attaches its callbacks to the lists below when the web layer is assembled.
# A hook call never raises and never changes engine behavior; with no
# subscribers it is a full no-op (the bot process runs without the web
# layer and without prometheus-client).

STAGE_HOOKS: list = []     # (stage: str, seconds: float) - per-stage search timings
EXTERNAL_HOOKS: list = []  # (call: str, outcome: str, seconds: float) - external calls
GAUGE_HOOKS: list = []     # (name: str, delta: float) - gauge increments/decrements


def emit_stage(stage: str, seconds: float) -> None:
    """Search stage finished (llm1_extract / yt_search / ...) in seconds."""
    for hook in list(STAGE_HOOKS):
        try:
            hook(stage, seconds)
        except Exception:
            pass


def emit_external(call: str, outcome: str, seconds: float) -> None:
    """External call (llm / lastfm / ytdlp / ...) finished with the given outcome."""
    for hook in list(EXTERNAL_HOOKS):
        try:
            hook(call, outcome, seconds)
        except Exception:
            pass


def emit_gauge(name: str, delta: float) -> None:
    """Gauge delta (e.g. +1/-1 of waiting for a yt-dlp slot)."""
    for hook in list(GAUGE_HOOKS):
        try:
            hook(name, delta)
        except Exception:
            pass


# Breaker registry for observation (the msm_circuit_open gauge reads states
# on scrape): name -> the most recently created breaker of the process.
BREAKERS: dict[str, "CircuitBreaker"] = {}


# ---------- circuit breaker ----------

class CircuitBreaker:
    """Consecutive-failure breaker: N failures in a row -> open (fast-fail without
    the call), after cooldown -> half-open (one probe). Success resets the counter."""

    def __init__(self, name: str, threshold: int = 3, cooldown: float = 60.0):
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at = 0.0
        # wave 45-3: registration for tech metrics (observation, not logic)
        BREAKERS[name] = self

    @property
    def is_open(self) -> bool:
        if self._failures < self.threshold:
            return False
        # half-open: cooldown expired — allow one probe (it closes or re-arms the circuit)
        return time.monotonic() - self._opened_at < self.cooldown

    def record_success(self) -> None:
        if self._failures:
            _log.info("circuit %s: closed (recovered after %d failures)", self.name, self._failures)
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()
            _log.warning("circuit %s: OPEN after %d consecutive failures (cooldown %.0fs)",
                         self.name, self._failures, self.cooldown)


# ---------- PII redaction ----------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s()-]{7,}\d)")  # 9+ digits with separators


def redact_for_log(text: str, max_len: int = 120) -> str:
    """Scrub a user description for the log: email and phone -> <redacted>, truncate.

    The request itself is untouched — only the log copy. A long description is cut
    at max_len (word boundaries don't matter: it's a log, not a UI)."""
    if not text:
        return ""
    s = _EMAIL_RE.sub("<redacted>", text)
    s = _PHONE_RE.sub("<redacted>", s)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s
