import sqlite3, json, time
con = sqlite3.connect("/app/data/music-search.db")
con.row_factory = sqlite3.Row
c = con.cursor()

print("=== TOTALS ===")
q = c.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
f = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
print(f"queries: {q}, feedback: {f}")

print("\n=== BY INTENT ===")
for r in c.execute("SELECT COALESCE(intent,'?') intent, COUNT(*) n FROM queries GROUP BY intent ORDER BY n DESC"):
    print(f"  {r['intent']}: {r['n']}")

print("\n=== BY CONFIDENCE ===")
for r in c.execute("SELECT COALESCE(confidence,'?') confidence, COUNT(*) n FROM queries GROUP BY confidence ORDER BY n DESC"):
    print(f"  {r['confidence']}: {r['n']}")

print("\n=== FEEDBACK ===")
for r in c.execute("SELECT feedback, COUNT(*) n FROM feedback GROUP BY feedback"):
    print(f"  {r['feedback']}: {r['n']}")

print("\n=== RECENT QUERIES (last 40) ===")
rows = c.execute("""SELECT q.id, q.created_at, q.description, q.intent, q.confidence,
                           q.best_title, q.best_score,
                           (SELECT GROUP_CONCAT(f.feedback) FROM feedback f WHERE f.query_id=q.id) AS fb
                    FROM queries q ORDER BY q.created_at DESC LIMIT 40""").fetchall()
for r in rows:
    ts = time.strftime('%m-%d %H:%M', time.localtime(r['created_at']))
    fb = r['fb'] or '-'
    print(f"[{ts}] intent={r['intent']} conf={r['confidence']} score={r['best_score']} fb={fb}")
    print(f"    Q: {r['description'][:120]}")
    print(f"    best: {(r['best_title'] or '')[:120]}")

# Hit-rate broken down by match_score and by auto/preview decision.
# Accumulates as "correct/incorrect" feedback is labeled; the goal is 100-200 queries per threshold bucket.
print("\n=== MATCH_SCORE vs HIT (F2+D3) ===")
rows = c.execute("""SELECT q.match_score, q.auto_download, q.ambiguous,
                           (SELECT GROUP_CONCAT(f.feedback) FROM feedback f WHERE f.query_id=q.id) AS fb
                    FROM queries q WHERE q.match_score IS NOT NULL""").fetchall()

def _hit(fb):
    return "hit" in (fb or "")

buckets = [(0.0, 0.40, "<0.40"), (0.40, 0.65, "0.40-0.65"),
           (0.65, 0.70, "0.65-0.70"), (0.70, 1.01, ">=0.70")]
print(f"  {'bucket':<12} {'n':>4} {'hits':>5} {'hit%':>7}")
for lo, hi, label in buckets:
    sel = [r for r in rows if lo <= (r["match_score"] or 0) < hi]
    if sel:
        hits = sum(1 for r in sel if _hit(r["fb"]))
        print(f"  {label:<12} {len(sel):>4} {hits:>5} {hits/len(sel)*100:>6.1f}%")
    else:
        print(f"  {label:<12}    0     0       -")

print("\n  === AUTO vs PREVIEW ===")
for ad, label in [(1, "auto"), (0, "preview")]:
    sel = [r for r in rows if r["auto_download"] == ad]
    if sel:
        hits = sum(1 for r in sel if _hit(r["fb"]))
        print(f"  {label:<8} n={len(sel):>3}  hits={hits:>3}  hit%={hits/len(sel)*100:>5.1f}%")
    else:
        print(f"  {label:<8} n=0")

# Why preview happens: some queries are ambiguous (close second candidate / low confidence),
# some fall below the auto threshold on score.
preview = [r for r in rows if r["auto_download"] == 0]
amb = sum(1 for r in preview if r["ambiguous"])
print(f"\n  preview всего={len(preview)}, из них по неоднозначности (ambiguous)={amb}")
