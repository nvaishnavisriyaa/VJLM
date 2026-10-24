"""Is the knowledge base at fault for a miss, or is retrieval?

diagnose_misses.py and why_miss.py both start from the gold chunk and ask where
it ranked. Both therefore assume the chunk is fine. This asks the prior
question: for a question we get wrong, does a WELL-FORMED passage holding the
answer exist in the corpus at all?

One pass over the enriched corpus, collecting every chunk that matches all of a
question's gold patterns, then classifying what it finds:

  ABSENT  nothing in 105k chunks matches  -> the crawl or the extractor lost it,
          or the answer was never on the site. No retrieval change can fix it.
  BLOB    the match is one huge chunk and its source has almost no others -> the
          page was never split, so the answer is diluted across thousands of
          characters and the embedding describes the page, not the clause.
  TABLE   digit- and delimiter-dense with little prose -> a PDF table flattened
          into unreadable text. Retrieval may find it; the model cannot read it.
  OK      a clean, ordinary passage holding the answer -> retrieval ranked it
          below 5 and the knowledge base is blameless.

    python evaluation/audit_corpus.py [qid qid ...]
"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402

ENRICHED = os.path.join(ROOT, "preprocessing", "out", "rag_chunks_enriched.jsonl")
# The 17 that survive the focus fix, i.e. genuine retrieval misses at topk=5.
DEFAULT = ("k01 d02 h09 h17 n03 n04 n05 n07 n14 n19 n23 n24 n25 n27 n31 n32 n34").split()
BLOB_CHARS = 4000


def shape(text):
    """Prose, or a flattened table?"""
    if not text:
        return "empty", 0.0
    digits = sum(c.isdigit() for c in text) / len(text)
    delims = (text.count("|") + text.count("\t")) / max(len(text.split()), 1)
    # A sentence needs a verb-ish run of lowercase words; tables have none.
    sents = len(re.findall(r"[a-z]{3,}\s+[a-z]{3,}\s+[a-z]{3,}", text))
    words = max(len(text.split()), 1)
    if digits > 0.18 or delims > 0.25 or sents / words < 0.02:
        return "table", digits
    return "prose", digits


def main():
    want = sys.argv[1:] or DEFAULT
    byid = {q[0]: q for q in Q1 + Q2}
    missing = [q for q in want if q not in byid]
    if missing:
        sys.exit("unknown qid: %s" % " ".join(missing))

    probes = []
    for qid in want:
        _, _, _, q, allx, src = byid[qid]
        probes.append((qid, q, [re.compile(a, re.I) for a in allx],
                       re.compile(src, re.I) if src else None, []))

    src_counts = {}
    n = 0
    with open(ENRICHED, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            n += 1
            sid = row.get("source_id") or row.get("url") or ""
            src_counts[sid] = src_counts.get(sid, 0) + 1
            text = row.get("text") or ""
            url = row.get("url") or ""
            for qid, q, pats, srcpat, found in probes:
                if srcpat and not srcpat.search(url):
                    continue
                if all(p.search(text) for p in pats):
                    if len(found) < 40:
                        found.append(row)
    print("scanned %d chunks\n" % n)

    verdicts = {}
    for qid, q, pats, srcpat, found in probes:
        if not found:
            verdicts[qid] = "ABSENT"
            print("%-4s ABSENT  no chunk matches | %s" % (qid, q[:62]))
            continue
        # Judge the best candidate: the shortest match is the most focused one.
        found.sort(key=lambda r: len(r.get("text") or ""))
        best = found[0]
        sid = best.get("source_id") or best.get("url") or ""
        sibs = src_counts.get(sid, 0)
        sh, dig = shape(best.get("text") or "")
        size = len(best.get("text") or "")
        if sh == "table":
            v = "TABLE"
        elif size >= BLOB_CHARS and sibs <= 3:
            v = "BLOB"
        else:
            v = "OK"
        verdicts[qid] = v
        print("%-4s %-6s %2d match, best %5dc, source has %4d chunks, %s | %s"
              % (qid, v, len(found), size, sibs, sh, q[:52]))
        print("      %s" % (best.get("url") or "")[:110])
        snip = re.sub(r"\s+", " ", (best.get("text") or ""))[:200]
        print("      %s" % snip)

    print("\n--- verdict")
    order = ["ABSENT", "BLOB", "TABLE", "OK"]
    for v in order:
        ids = [q for q in want if verdicts[q] == v]
        if ids:
            print("%-6s %2d/%d  %s" % (v, len(ids), len(want), " ".join(ids)))
    kb = sum(1 for q in want if verdicts[q] in ("ABSENT", "BLOB", "TABLE"))
    print("\nknowledge base at fault: %d of %d misses" % (kb, len(want)))


if __name__ == "__main__":
    main()
