"""Sonic mode eval: for reference pairs, checks that the sonic answer is in the top 3.

A live run of the full pipeline (LLM bothub + Last.fm getSimilar + yt-dlp), not a mock.
Requires BOTHUB_API_KEY and LASTFM_API_KEY in .env (keys are kept out of the repo).
Each pair makes several network calls, minutes per pair in total; the cost is pennies.

Run from the repo root:  python scripts/sonic_eval.py

A pair = (a "similar to ..." query, a list of acceptable sonic artists). Hit: at
least one acceptable artist appears in the channel OR the title of one of the top-3
candidates. An artist, not a specific track: a "sonic answer" is a sound-alike
neighbor, while the reference track (e.g. Poor Man's Moody Blues) is a bonus,
not a criterion.

Pure "sounds like" pairs are rare: a clone (Greta Van Fleet) and an explicit
homage (Barclay James Harvest). "Influenced by" (Pixies→Nirvana, Muse→Radiohead)
is not "sounds like": the acceptable answer there is genre neighbors, hence the
wider list of acceptable artists.
"""
import asyncio
import sys
from pathlib import Path

# put the repo root on sys.path: `python scripts/sonic_eval.py` puts only scripts/ on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from services.llm_search import BothubClient, search_candidates, _ref_key, _channel_key
from services.metadata import MetadataService

# reference pairs: (query, [acceptable sonic artists])
PAIRS = [
    ("похоже на The Moody Blues — Nights in White Satin", ["Barclay James Harvest"]),
    ("похоже на Led Zeppelin", ["Greta Van Fleet"]),
    ("похоже на Nirvana", ["Soundgarden", "Alice In Chains", "Pearl Jam"]),
    ("похоже на Radiohead", ["Muse", "Grizzly Bear", "Other Lives"]),
]


def is_hit(c, artist_keys: list[str]) -> bool:
    """An acceptable artist appears in the candidate's channel or title."""
    ch = _channel_key(c.channel)
    t = _ref_key(c.title)
    return any(k in ch or k in t for k in artist_keys if k)


async def run_pair(bothub, meta, query: str, expected_artists: list[str]):
    out = await search_candidates(query, bothub, metadata_svc=meta)
    artist_keys = [_ref_key(a) for a in expected_artists]
    top3 = out.candidates[:3]
    hit = any(is_hit(c, artist_keys) for c in top3)
    return hit, top3, out.intent, out.need_clarify, artist_keys


async def main() -> None:
    cfg = load_config()
    bothub = BothubClient(cfg.bothub_base, cfg.bothub_api_key, cfg.llm_model)
    meta = MetadataService(cfg.lastfm_api_key)
    passed = 0
    for query, expected_artists in PAIRS:
        print(f"\n=== {query!r}  (expecting {', '.join(expected_artists)})")
        try:
            hit, top3, intent, clarify, artist_keys = await run_pair(
                bothub, meta, query, expected_artists)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print(f"  intent={intent} clarify={clarify}")
        for i, c in enumerate(top3, 1):
            mark = "  <== HIT" if is_hit(c, artist_keys) else ""
            print(f"  {i}. {c.channel} — {c.title}  [score={c.score:.2f} match={c.match_score:.2f}]{mark}")
        if hit:
            passed += 1
        else:
            print("  (sonic answer is NOT in top-3)")
    print(f"\n=== {passed}/{len(PAIRS)} pairs: sonic answer in top-3 ===")


if __name__ == "__main__":
    asyncio.run(main())
