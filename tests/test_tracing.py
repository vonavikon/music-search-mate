"""Langfuse tracing. Off by default without keys, usage mapping, score guards.

The client is injected via tracing.init(_inject=...) — no network or SDK involved.
"""
import pytest

from services import tracing
from services.tracing import SearchTrace, usage_details


class FakeObs:
    def __init__(self, parent, name, as_type="span", input=None, model=None, **kw):
        self.parent = parent
        self.name = name
        self.as_type = as_type
        self.input = input
        self.model = model
        self.finished = None

    def update(self, output=None, **kw):
        self.finished = {"output": output, **kw}

    def end(self):
        self.ended = True


class FakeRoot:
    def __init__(self):
        self.children = []
        self.updated = {}

    def start_observation(self, **kw):
        child = FakeObs(self, **kw)
        self.children.append(child)
        return child

    def update(self, **kw):
        self.updated.update(kw)


class FakeClient:
    """Stub client: start_observation returns a FakeRoot, create_score appends to a list."""

    def __init__(self):
        self.traces = []
        self.scores = []

    def start_observation(self, **kw):
        root = FakeRoot()
        self.traces.append((kw, root))
        return root

    def create_score(self, **kw):
        self.scores.append(kw)


@pytest.fixture(autouse=True)
def reset_tracing():
    yield
    tracing.shutdown()


def test_off_without_keys():
    assert tracing.init() is False
    t = SearchTrace("a" * 32, 1, 2, "desc")
    # no-op: the methods do not fail and create nothing
    assert t.span("x") is not None
    t.span("x").finish(output={})
    t.tag(["similar"])
    t.finish(output={})
    tracing.score_feedback("a" * 32, True, 0, "t")
    assert tracing._client is None


def test_usage_details_mapping():
    assert usage_details({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}) == \
        {"input": 10, "output": 5, "total": 15}
    # newer OpenAI naming
    assert usage_details({"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}) == \
        {"input": 3, "output": 4, "total": 7}
    assert usage_details(None) is None
    assert usage_details({"foo": 1}) is None


def test_trace_hierarchy_with_fake_client():
    fake = FakeClient()
    assert tracing.init(_inject=fake) is True
    t = SearchTrace("ab" * 16, 42, 100500, "ищу как Queen")
    assert fake.traces, "root observation создан"
    kw, root = fake.traces[0]
    assert kw["name"] == "search"
    # pipeline: generation + span + finalization
    gen = t.span("llm-extract", input="q", as_type="generation", model="m1")
    gen.finish(output={"intent": "similar"}, usage_details={"input": 10, "output": 5, "total": 15})
    sp = t.span("youtube-search", input={"queries": ["a"]})
    sp.finish(output={"raw": 7})
    t.tag(["similar"])
    t.finish(output={"decision": "preview"})
    kinds = [(c.name, c.as_type) for c in root.children]
    assert ("llm-extract", "generation") in kinds and ("youtube-search", "span") in kinds
    gen_fin = root.children[0].finished
    assert gen_fin["usage_details"] == {"input": 10, "output": 5, "total": 15}
    assert root.updated.get("metadata") == {"tags": ["similar"]}


def test_score_feedback_guard_and_value():
    fake = FakeClient()
    tracing.init(_inject=fake)
    tracing.score_feedback("c" * 32, True, 0, "Track X")
    tracing.score_feedback("d" * 32, False, 2, "Track Y")
    assert fake.scores[0]["value"] == 1.0 and fake.scores[0]["data_type"] == "BOOLEAN"
    assert fake.scores[1]["value"] == 0.0
    # legacy correlation_id format (str(user.id)) — score is not sent
    tracing.score_feedback("123456789", True, 0, "old")
    tracing.score_feedback(None, True, 0, "none")
    assert len(fake.scores) == 2


@pytest.mark.asyncio
async def test_chat_json_records_generation(bothub_client):
    """chat_json with a trace records a generation with usage from bothub (last_usage is finally read)."""
    fake = FakeClient()
    tracing.init(_inject=fake)
    t = SearchTrace("e" * 32, 1, 2, "desc")
    _, root = fake.traces[0]

    async def fake_post(url, json=None, headers=None):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": '{"ok": 1}'}}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
        return R()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bothub_client._http, "post", fake_post)
        res = await bothub_client.chat_json("s", "u", trace=t, gen_name="llm-extract")
    assert res == {"ok": 1}
    gens = [c for c in root.children if c.as_type == "generation"]
    assert len(gens) == 1 and gens[0].name == "llm-extract"
    assert gens[0].finished["usage_details"] == {"input": 7, "output": 3, "total": 10}


@pytest.mark.asyncio
async def test_chat_json_error_marks_level_error(bothub_client):
    """A failed LLM call closes the generation with level=ERROR, enabling an errors-only filter."""
    fake = FakeClient()
    tracing.init(_inject=fake)
    t = SearchTrace("f" * 32, 1, 2, "desc")
    _, root = fake.traces[0]

    async def boom(url, json=None, headers=None):
        raise RuntimeError("network down")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bothub_client._http, "post", boom)
        res = await bothub_client.chat_json("s", "u", trace=t, gen_name="llm-extract")
    assert res is None
    gen = root.children[0]
    assert gen.finished["level"] == "ERROR"
    assert "network down" in gen.finished["output"]["error"]
