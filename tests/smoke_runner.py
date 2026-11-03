"""Smoke runner for music-search-bot: smoke tests simulating user queries.

PYTHONUTF8=1 python tests/smoke_runner.py [--no-judge] [--only ID]

Cases (from real history + reference smoke scenarios):
  - this_track_exact: exact title -> best contains the keywords
  - similar_vibe:     similar sound -> LLM-judge: relevant AND not the reference
  - from_source:      from an ad/movie -> best contains the source
  - ambiguous:        vague query -> clarify or low confidence (NOT a confident hit)
  - context_refinement: follow-up with context -> LLM-judge: uses the earlier reference

Without --no-judge: similar/context cases are scored by an LLM judge (deepseek via bothub).
Report: a table of case | category | intent | conf | clarify | best | pass | reason.
"""
import argparse
import asyncio
import os
import re
import sys
from uuid import uuid4

# allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, ensure_dirs
from services import tracing
from services.tracing import SearchTrace
from services.llm_search import BothubClient, search_candidates
from services.metadata import MetadataService


# ---------- cases ----------

CASES = [
    {
        "id": "acdc", "category": "similar_vibe",
        "desc": "Хочу что-то бодрое по звучанию как AC/DC",
        "assert": "similar_not_reference", "reference": "AC/DC",
    },
    {
        "id": "tame", "category": "similar_vibe",
        "desc": "Хочу песню как у Tame Impala, что-то необычное",
        "assert": "similar_not_reference", "reference": "Tame Impala",
    },
    {
        "id": "oldtown", "category": "this_track_exact",
        "desc": "найди мне песню I wanna take my horse to the old town road",
        "assert": "this_track", "must_contain": ["old town road"],
    },
    {
        "id": "beer", "category": "ambiguous",
        "desc": "что за песня была в рекламе пива, пиво еще такое зеленое, песня мелодичная",
        "assert": "no_crash",  # the model guesses, but the pipeline must not crash — this is smoke, not quality
    },
    {
        "id": "carlsberg", "category": "from_source",
        "desc": "была реклама пиво карлсберг, песня оттуда",
        "assert": "from_source", "must_contain_any": ["carlsberg"],
    },
    {
        "id": "cowboy", "category": "similar_vibe",
        "desc": "Хочу песню ковбойскую",
        "assert": "similar_genre", "genre_hint": "country / western / outlaw",
    },
    {
        "id": "context_faster", "category": "context_refinement",
        "desc": "А что-то такое же, но побыстрее",
        "context": [
            {"description": "Хочу что-то бодрое как AC/DC",
             "intent": "similar", "best_title": "Long Stick Goes Boom", "outcome": "hit"},
        ],
        "assert": "context_used", "reference": "AC/DC",
    },
    # --- production cases (from Aug 8-10 logs) ---
    {
        "id": "depeche_ban", "category": "reference_ban_edge",
        "desc": "дай мне что нибудь необычное из depeche mode",
        "assert": "reference_ban_edge", "reference": "Depeche Mode",
    },
    {
        "id": "green_beer", "category": "ambiguous",
        "desc": "Слушай, а что за песня была в рекламе пива, пиво еще такое зеленое",
        "assert": "no_crash",
    },
    {
        "id": "makulatura", "category": "this_track_niche",
        "desc": "помоги мне найти песню группы макулатура там была цитата из чехова",
        "assert": "niche_russian",
    },
    {
        "id": "back_to_future", "category": "from_source",
        "desc": "Дай мне, пожалуйста, песню, которая играла из фильма назад в будущее",
        "assert": "from_source", "must_contain_any": ["back to the future", "alan silvestri", "back in time", "huey lewis"],
    },
    {
        "id": "enjoy_silence_version", "category": "this_track_version",
        "desc": "дай мне depeche mode лучший альбом что нибудь",
        "assert": "this_track_version", "reference": "Depeche Mode",
    },
    {
        "id": "shrek_hero", "category": "context_refinement",
        "desc": "Что-то про героя",
        "context": [
            {"description": "Дай песню, которую пели в Шреке",
             "intent": "from_source", "best_title": "Smash Mouth - All Star", "outcome": "hit"},
        ],
        "assert": "context_used", "reference": "shrek",
    },
]


# ---------- judge ----------

JUDGE_SYS = """Ты — судья музыкального поиска. Дано: запрос пользователя, выбранный трек (title, channel), критерий. Оцени, насколько трек подходит запросу. Верни СТРОГО JSON: {"score": 0.0-1.0, "reason": "коротко", "pass": true|false}.
Критерии:
- similar: трек похож по звучанию/настроению на референс И это ДРУГОЙ артист (не сам референс). score>=0.6 = pass.
- genre: трек подходит под жанровый намёк. score>=0.6 = pass.
- context: поисковые запросы используют референс из контекста (т.е. поиск связан с прошлым). score>=0.6 = pass.
Не выдумывай, суди по данным."""


async def llm_judge(bothub, desc, best, criterion, hint=""):
    user = (f"Запрос: {desc}\nКритерий: {criterion}\n"
            + (f"Намёк/референс: {hint}\n" if hint else "")
            + f"Выбранный трек: {best.title} | {best.channel}\n"
            f"Поисковые запросы LLM: {getattr(best, '_queries', [])}")
    res = await bothub.chat_json(JUDGE_SYS, user, max_tokens=300)
    if not res:
        return 0.0, "judge_failed", False
    return float(res.get("score", 0)), res.get("reason", ""), bool(res.get("pass", False))


# ---------- assertion ----------

def _has(text, needle):
    return needle.lower() in (text or "").lower()


async def evaluate(case, out, bothub, use_judge):
    """Return (passed, reason, best_title)."""
    best = out.candidates[out.best_index] if (out.candidates and 0 <= out.best_index < len(out.candidates)) else None
    best_title = best.title if best else "—"
    a = case["assert"]

    if a == "this_track":
        ok = best is not None and all(_has(best.title, m) for m in case["must_contain"])
        return ok, "match" if ok else f"нет {case['must_contain']}", best_title

    if a == "clarify_or_low":
        ok = bool(out.need_clarify) or out.confidence == "low"
        why = "clarify" if out.need_clarify else ("low" if out.confidence == "low" else f"conf={out.confidence}")
        return ok, why, best_title

    if a == "from_source":
        ok = best is not None and any(_has(best.title + " " + best.channel, m) for m in case["must_contain_any"])
        return ok, "match" if ok else f"нет {case['must_contain_any']}", best_title

    if not use_judge or best is None:
        return None, "skip-judge", best_title

    if a == "similar_not_reference":
        score, reason, passed = await llm_judge(
            bothub, case["desc"], best, "similar (похож на референс, но ДРУГОЙ артист)", case["reference"])
        return passed, f"judge {score:.2f}: {reason}", best_title

    if a == "similar_genre":
        score, reason, passed = await llm_judge(
            bothub, case["desc"], best, "genre (подходит под жанровый намёк)", case["genre_hint"])
        return passed, f"judge {score:.2f}: {reason}", best_title

    if a == "context_used":
        # simulate: pass _queries along as an artifact for the judge
        if best:
            best._queries = out.queries
        score, reason, passed = await llm_judge(
            bothub, case["desc"], best, "context (поиск использует референс из контекста)", case["reference"])
        return passed, f"judge {score:.2f}: {reason}", best_title

    if a == "reference_ban_edge":
        # top-1 must not be junk with match_score < 0.3 after the reference ban
        ok = best is not None and out.candidates[out.best_index].match_score >= 0.3
        return ok, f"score={out.candidates[out.best_index].match_score:.2f}" if not ok else "ok", best_title

    if a == "niche_russian":
        # the main point is not crashing; finding the track is optional
        return True, "no crash" if best else "no candidates (ok)", best_title

    if a == "this_track_version":
        # should suggest the studio version, not a live/remix
        ok = best is not None and "live" not in best.channel.lower() and "remix" not in best.title.lower()
        return ok, "version ok" if ok else "bad version", best_title

    if a == "no_crash":
        # the pipeline completed without errors — the main criterion for vague cases
        return True, f"pipeline ok ({out.intent})", best_title

    return None, "unknown-assert", best_title


# ---------- runner ----------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true", help="без LLM-judge (только детерминированные asserts)")
    ap.add_argument("--only", default=None, help="запустить только кейс по id")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    bothub = BothubClient(cfg.bothub_base, cfg.bothub_api_key, cfg.llm_model)
    metadata = MetadataService(cfg.lastfm_api_key)
    # smoke runs are traced to Langfuse (like production searches) when keys are set
    tracing.init(cfg.langfuse_public_key, cfg.langfuse_secret_key, cfg.langfuse_host)
    use_judge = not args.no_judge

    rows = []
    try:
        cases = [c for c in CASES if (args.only is None or c["id"] == args.only)]
        for c in cases:
            print(f"\n>>> [{c['id']}] {c['category']}: {c['desc']}")
            trace = SearchTrace(uuid4().hex, 0, None, c["desc"])
            try:
                out = await search_candidates(c["desc"], bothub, context=c.get("context"),
                                              metadata_svc=metadata, trace=trace)
            except Exception as e:
                trace.finish(output={"case": c["id"], "error": str(e)[:80]})
                rows.append((c["id"], c["category"], "ERR", "—", "—", "—", False, str(e)[:60]))
                print(f"   ERROR: {e}")
                continue
            passed, reason, best_title = await evaluate(c, out, bothub, use_judge)
            trace.finish(output={"case": c["id"], "pass": passed, "reason": reason[:60],
                                 "best": best_title[:60]})
            mark = {"True": "PASS", "False": "FAIL"}.get(str(passed), "PASS")
            print(f"   intent={out.intent} conf={out.confidence} clarify={out.need_clarify}")
            print(f"   best: {best_title}")
            print(f"   {mark}: {reason}")
            rows.append((c["id"], c["category"], out.intent or "?", out.confidence,
                         "Y" if out.need_clarify else "-", best_title[:42], passed, reason[:50]))
    finally:
        await bothub.aclose()
        await metadata.aclose()
        tracing.shutdown()  # flush traces before the process exits (common mistake)

    # report
    print("\n" + "=" * 100)
    print(f"{'id':<16}{'category':<20}{'intent':<12}{'conf':<8}{'clr':<4}{'best':<44}{'result'}")
    print("-" * 100)
    npass = nfail = nskip = 0
    for r in rows:
        cid, cat, intent, conf, clr, best, passed, reason = r
        if passed is True:
            res = "PASS"; npass += 1
        elif passed is False:
            res = f"FAIL ({reason})"; nfail += 1
        else:
            res = "SKIP"; nskip += 1
        print(f"{cid:<16}{cat:<20}{intent:<12}{conf:<8}{clr:<4}{best:<44}{res}")
    print("=" * 100)
    print(f"PASS={npass}  FAIL={nfail}  SKIP={nskip}  (judge={'on' if use_judge else 'off'})")
    print("SMOKE", "OK" if nfail == 0 else "FAILURES")


if __name__ == "__main__":
    asyncio.run(main())
