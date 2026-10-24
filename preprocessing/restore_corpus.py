"""Unpack the knowledge base that ships with the repository.

The corpus is versioned compressed. Uncompressed it is 125 MB + 129 MB, which
GitHub rejects (100 MB per file); gzipped it is ~28 MB each, which plain git
takes without LFS. The text is ~22% of its original size because chunk records
are highly repetitive JSON.

So a fresh clone has the knowledge base but not in the form the code reads.
This restores it:

    python preprocessing/restore_corpus.py

Then build the vector index - that is NOT versioned, because it is 519 MB of
embeddings regenerable from this corpus:

    python rag/index.py

Nothing here re-crawls or re-extracts. Run this instead of the ingestion
pipeline when you only want the knowledge base as it was last published.
"""
import gzip, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.environ.get("VJLM_OUT_DIR") or os.path.join(ROOT, "preprocessing", "out")
FILES = ["rag_chunks.jsonl", "rag_chunks_enriched.jsonl"]


def restore(name, force=False):
    dst = os.path.join(OUT_DIR, name)
    src = dst + ".gz"
    if not os.path.exists(src):
        print("missing  %s" % src)
        return False
    if os.path.exists(dst) and not force:
        # Don't clobber a corpus the local pipeline just rebuilt: that one is
        # newer than whatever was published.
        print("kept     %s (%.0f MB on disk already; --force to overwrite)"
              % (name, os.path.getsize(dst) / 1e6))
        return True
    tmp = dst + ".part"
    with gzip.open(src, "rb") as fh, open(tmp, "wb") as out:
        shutil.copyfileobj(fh, out, 1 << 20)
    os.replace(tmp, dst)
    print("restored %s (%.0f MB)" % (name, os.path.getsize(dst) / 1e6))
    return True


def main():
    force = "--force" in sys.argv
    ok = all(restore(n, force) for n in FILES)
    if not ok:
        sys.exit("corpus incomplete - run the ingestion pipeline instead")
    print("next: python rag/index.py")


if __name__ == "__main__":
    main()
