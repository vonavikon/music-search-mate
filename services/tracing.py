"""Langfuse tracing of the search pipeline (v1.10).

One search = one trace: trace_id = correlation_id (uuid4.hex), also written to
SQLite and structured logs — the "log <-> DB <-> trace" link.

Hierarchy (langfuse skill, references/instrumentation.md):
  root span "search"
    ├─ generation "llm-extract"   (model, usage, input=query, output=JSON extract)
    ├─ retriever "reference-resolve"  (similar-fix: Last.fm/yt-dlp reference resolve)
    ├─ span "youtube-search"      (queries -> raw count)
    ├─ span "filter-enrich"       (raw -> filtered/enriched)
    ├─ generation "llm-rank"      (model, usage, input=candidates, output=JSON rank)
    └─ span "decide"              (auto/preview, match_score)
"Hit / miss" feedback -> score "user_feedback" on the trace (hit=1/miss=0).

Off without keys — the LASTFM_API_KEY pattern: empty env = feature silently off.
The SDK import is lazy, after load_dotenv (skill: import after env).
Tracing must not break search: all calls are wrapped, exceptions swallowed with
a warning.
PII: description goes through redact_for_log (email/phone).
"""
import logging

from services.observability import redact_for_log

_log = logging.getLogger("bot.tracing")

_client = None
_enabled = False
_environment = ""  # production/development (best practice: test traces stay out of prod dashboards)


def init(public_key: str = "", secret_key: str = "", host: str = "",
         environment: str = "", _inject=None) -> bool:
    """Initialization. Empty keys/host = tracing off. Returns on/off.
    _inject substitutes a client in tests (no network and no SDK)."""
    global _client, _enabled, _environment
    if _enabled:
        return True
    _environment = (environment or "").strip()
    if _inject is not None:
        _client, _enabled = _inject, True
        return True
    if not (public_key and secret_key and host):
        _log.info("langfuse tracing off (no keys)")
        return False
    try:
        from langfuse import Langfuse  # lazy: after load_dotenv
    except Exception as e:  # SDK missing/broken — the bot runs without tracing
        _log.warning("langfuse SDK unavailable (%s) — tracing off", e)
        return False
    try:
        _client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        _enabled = True
        _log.info("langfuse tracing on (host=%s, env=%s)", host, _environment or "default")
    except Exception as e:
        _log.warning("langfuse init failed: %s — tracing off", e)
        _client = None
    return _enabled


def shutdown() -> None:
    """Flush on bot shutdown: without it the tail of traces never arrives (skill: common mistake)."""
    global _client, _enabled
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:
            pass
    _client, _enabled = None, False


def score_feedback(trace_id: str | None, hit: bool, rank: int, title: str) -> None:
    """The "hit/miss" button -> a score on the trace. Old correlation_ids (= str(user_id),
    pre-v1.10) are rejected by length: no trace with such an id exists."""
    if not (_enabled and _client and trace_id and len(trace_id) == 32):
        return
    try:
        _client.create_score(
            trace_id=trace_id, name="user_feedback",
            value=1.0 if hit else 0.0, data_type="BOOLEAN",
            comment=f"{'hit' if hit else 'miss'} rank={rank} {title[:80]}",
        )
    except Exception as e:
        _log.warning("langfuse score failed: %s", e)


class _Obs:
    """A child observation with safe completion. finish() takes output and
    optionally usage_details/model for generations (usage is known after the call)."""

    __slots__ = ("_o",)

    def __init__(self, parent, name: str, as_type: str = "span", input=None,
                 model: str | None = None, model_parameters: dict | None = None):
        self._o = None
        if parent is None:
            return
        try:
            kw = {"name": name, "as_type": as_type, "input": input}
            if model:
                kw["model"] = model
            if model_parameters:
                kw["model_parameters"] = model_parameters
            self._o = parent.start_observation(**kw)
        except Exception as e:
            _log.warning("langfuse observation '%s' failed: %s", name, e)

    def finish(self, output=None, usage_details: dict | None = None, level: str | None = None) -> None:
        if self._o is None:
            return
        try:
            kw = {"output": output}
            if usage_details:
                kw["usage_details"] = usage_details
            if level:
                # "ERROR" marks the failed stage — the "problems only" filter in the UI
                kw["level"] = level
            self._o.update(**kw)
            self._o.end()
        except Exception as e:
            _log.warning("langfuse observation finish failed: %s", e)


class SearchTrace:
    """A single search request. Without init() all methods are no-op: pipeline
    call sites are unconditional, tracing must not affect the search."""

    def __init__(self, trace_id: str, user_id: int, chat_id: int | None, description: str):
        self.trace_id = trace_id
        self.user_id = user_id
        self.chat_id = chat_id
        self.description = description
        self._ctx = None
        self._root = None
        if not _enabled:
            return
        try:
            from langfuse import propagate_attributes
            from langfuse.types import TraceContext
            # user_id/session_id on the trace (user -> per-user filters and cost,
            # chat_id -> Sessions grouping of the bot dialog)
            self._ctx = propagate_attributes(
                user_id=str(user_id),
                session_id=f"chat-{chat_id}" if chat_id is not None else None,
                environment=(_environment or None),
            )
            self._ctx.__enter__()
            self._root = _client.start_observation(
                trace_context=TraceContext(trace_id=trace_id),
                name="search", input=redact_for_log(description, max_len=400),
            )
        except Exception as e:
            _log.warning("langfuse trace start failed: %s", e)
            self._ctx = self._root = None

    def span(self, name: str, input=None, as_type: str = "span",
             model: str | None = None, model_parameters: dict | None = None) -> _Obs:
        """A child observation. Create it BEFORE the step (timing), finish() after."""
        return _Obs(self._root, name, as_type=as_type, input=input,
                    model=model, model_parameters=model_parameters)

    def tag(self, tags: list[str]) -> None:
        """Append tags to the trace (intent after extract) via root metadata."""
        if self._root is None or not tags:
            return
        try:
            self._root.update(metadata={"tags": tags})
        except Exception:
            pass

    def finish(self, output=None) -> None:
        """Close the trace: root output = the pipeline's final decision."""
        if self._root is not None:
            try:
                self._root.update(output=output)
                self._root.end()
            except Exception as e:
                _log.warning("langfuse trace finish failed: %s", e)
            self._root = None
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._ctx = None


def usage_details(usage: dict | None) -> dict | None:
    """OpenAI usage (bothub) -> langfuse usage_details {input, output, total}.
    Tolerates both field namings (prompt/completion and input/output)."""
    if not isinstance(usage, dict):
        return None
    inp = usage.get("prompt_tokens", usage.get("input_tokens"))
    out = usage.get("completion_tokens", usage.get("output_tokens"))
    tot = usage.get("total_tokens")
    d = {}
    if isinstance(inp, int):
        d["input"] = inp
    if isinstance(out, int):
        d["output"] = out
    if isinstance(tot, int):
        d["total"] = tot
    return d or None
