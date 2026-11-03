"""CircuitBreaker and redact_for_log tests."""
from services.observability import CircuitBreaker, redact_for_log


# ════ circuit breaker ════

def test_breaker_opens_after_threshold():
    cb = CircuitBreaker("test", threshold=3, cooldown=60.0)
    assert not cb.is_open

    cb.record_failure()
    cb.record_failure()
    assert not cb.is_open  # below threshold yet

    cb.record_failure()
    assert cb.is_open


def test_breaker_half_open_after_cooldown(monkeypatch):
    t = [100.0]
    monkeypatch.setattr("services.observability.time.monotonic", lambda: t[0])
    cb = CircuitBreaker("test", threshold=2, cooldown=60.0)

    cb.record_failure()
    cb.record_failure()
    assert cb.is_open

    t[0] = 161.0  # cooldown expired
    assert not cb.is_open  # half-open: a single probe


def test_breaker_success_resets(monkeypatch):
    t = [100.0]
    monkeypatch.setattr("services.observability.time.monotonic", lambda: t[0])
    cb = CircuitBreaker("test", threshold=2, cooldown=60.0)

    cb.record_failure()
    cb.record_failure()
    assert cb.is_open

    cb.record_success()
    assert not cb.is_open
    # after reset, N consecutive failures are needed to open again
    cb.record_failure()
    assert not cb.is_open


# ════ PII redaction ════

def test_redact_email():
    assert "user@example.com" not in redact_for_log("пиши на user@example.com плиз")
    assert "<redacted>" in redact_for_log("мой email a.b-c+1@sub.domain.org ок")


def test_redact_phone():
    assert "+7 900 123-45-67" not in redact_for_log("позвони +7 900 123-45-67")
    assert "89001234567" not in redact_for_log("телефон 89001234567")


def test_redact_truncates_long_text():
    long = "слово " * 100
    out = redact_for_log(long, max_len=60)
    assert len(out) <= 60
    assert out.endswith("…")


def test_redact_empty_and_none():
    assert redact_for_log("") == ""
    assert redact_for_log(None) == ""


def test_redact_preserves_plain_text():
    assert redact_for_log("рок как AC/DC") == "рок как AC/DC"
