"""VJLM web app - local CPU inference, role-aware retrieval, cited answers.

Auth is a Phase-1 role simulation. The RBAC gate it feeds is real: retrieval
filters by visibility/audience BEFORE anything reaches the prompt builder.
"""
import json, os, re, sys, time, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, Response, render_template, stream_with_context
from rag.answer import attribute, with_supplement, used_from, rescue
from rag.retrieve import Retriever, build_prompt
from ingestion.refresh import Refresher

MODEL_DIR = os.path.join(ROOT, "quantization", "builds")
app = Flask(__name__)

_ret = None
_llm = None
_lock = threading.Lock()
_ret_lock = threading.Lock()


def pick_model():
    """Prefer the fine-tuned VJLM build; fall back to the base model."""
    for name in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []:
        if name.startswith("vjlm") and name.endswith(".gguf"):
            return os.path.join(MODEL_DIR, name), "fine-tuned"
    for name in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []:
        if name.endswith(".gguf"):
            return os.path.join(MODEL_DIR, name), "base"
    return None, None


INDEX_FILE = os.path.join(ROOT, "rag", "index", "bm25.pkl")
CHUNKS_FILE = os.path.join(ROOT, "preprocessing", "out", "rag_chunks.jsonl")


def get_retriever():
    """None until an index exists. A fresh clone has no rag/index/ (it is
    gitignored and rebuildable), and the app must still start and say so
    rather than crashing on import."""
    global _ret
    if _ret is None and os.path.exists(INDEX_FILE):
        with _ret_lock:
            if _ret is None:
                _ret = Retriever()
    return _ret


def reload_retriever():
    """Swap in a freshly built index. The new Retriever is constructed first
    and only then published, so in-flight queries keep using the old one
    instead of hitting a half-written index."""
    global _ret
    fresh = Retriever()
    with _ret_lock:
        _ret = fresh
    print("[app] retriever reloaded: %d chunks, dense=%s"
          % (len(fresh.meta), fresh.dense is not None), flush=True)


# The pipeline is driven by the app: starting VJLM is what keeps the knowledge
# base current. Phase 3's "automatic means no human step in steady state".
refresher = Refresher(on_index_rebuilt=reload_retriever)


_SRC_START = re.compile(r"\n\s*Sources?\s*:", re.I)


def safe_prefix_len(acc):
    """How much of the streamed text can be shown without risking the start of
    the model's own 'Source:' line (which is removed from the final answer)."""
    m = _SRC_START.search(acc)
    if m:
        return m.start()
    i = acc.rfind("\n")
    if i != -1 and "source".startswith(acc[i + 1:].strip().lower()[:6]):
        return i
    return len(acc)


LLAMA_URL = os.environ.get("VJLM_LLAMA_URL", "http://127.0.0.1:8080")


# Sampling temperature. 0.2 reads better for a person; an accuracy run needs
# the same answer every time, or a re-measurement shuffles one or two of 26
# questions (3.8% each) and a real change cannot be told from the sampler.
# Set VJLM_TEMP=0 for evaluation runs.
TEMP = float(os.environ.get("VJLM_TEMP", "0.2"))
SEED = int(os.environ.get("VJLM_SEED", "-1"))


def llama_chat(messages, max_tokens=420, temperature=None):
    """Talk to a local llama-server (official build, ~39x faster than the
    generic llama-cpp-python wheel on this CPU - see QUANT/RUNTIME bench).
    Local HTTP to 127.0.0.1 is not an external API."""
    import urllib.request, urllib.error
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": TEMP if temperature is None else temperature,
                       "seed": SEED, "stream": False}).encode()
    req = urllib.request.Request(LLAMA_URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()


def llama_stream(messages, max_tokens=420, temperature=None):
    """Yield answer text as llama-server produces it.

    At the measured CPU rates (14.7 tok/s Llama, 8.07 Mistral) a non-streamed
    answer is several seconds of blank screen, so the UI streams instead.
    """
    import urllib.request
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": TEMP if temperature is None else temperature,
                       "seed": SEED, "stream": True}).encode()
    req = urllib.request.Request(LLAMA_URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            piece = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if piece:
                yield piece


def llama_up():
    import urllib.request
    try:
        with urllib.request.urlopen(LLAMA_URL + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


# The page lives in app/templates/index.html with app/static/{app.css,app.js}.


@app.route("/")
def index():
    return render_template("index.html")


# Kept in sync with rag/retrieve.py WEIGHTS. An unlisted mode silently
# falls back to "hybrid", which would make a comparison look like it ran
# when it did not.
VALID_MODES = ("hybrid", "bm25", "dense", "coverage", "lexical", "hybrid3")
ABSTAIN = "I could not find this in the available VNRVJIET sources."


# Conversation is not a question. Without this, "hi" went to retrieval as a
# search query, came back with five irrelevant passages (a newsletter matching
# "hi"), and the fine-tuned model correctly refused - a right answer to a
# question nobody asked. Greetings, thanks and "what are you" get a direct
# reply: no retrieval, no sources, no model call.
_ABOUT = ("I'm VJLM (Vignana Jyothi Language Model), a Llama-3.2-3B model fine-tuned "
          "for VNRVJIET and running entirely on this computer's CPU, with no external AI "
          "service. I answer from documents the college publishes on its website: "
          "academic regulations, academic calendars, syllabi, notices and circulars, "
          "admissions and fees, and department, faculty and placement pages. Every "
          "answer cites its source, and if the documents don't contain the answer I "
          "say so instead of guessing.")
_TOPICS = ("academic regulations, the academic calendar, admissions and fees, "
           "syllabus, departments and faculty, placements, or campus life")
_SMALL_TALK = [
    (re.compile(r"^(hi+|hello+|hey+|hai|namaste|greetings|good\s+(morning|afternoon|evening))"
                r"(\s+(vjlm|there|bro|sir|madam|all))?[\s!.,?]*$", re.I),
     "Hi! I'm VJLM, VNRVJIET's campus assistant. Ask me about %s, and I'll answer "
     "from the college's own published documents, with the source for every answer." % _TOPICS),
    (re.compile(r"^(thanks?(\s+you)?|thank\s+you(\s+so\s+much)?|thx|ty|ok(ay)?|cool|great|nice|"
                r"got\s+it|bye|goodbye|see\s+you)[\s!.,?]*$", re.I),
     "You're welcome! Ask me anything else about VNRVJIET: %s." % _TOPICS),
    (re.compile(r"^((who|what)\s+(are|r)\s+(you|u)|what\s+(is|'s)\s+vjlm|what\s+can\s+you\s+do|"
                r"how\s+can\s+you\s+help(\s+me)?|help)[\s!.,?]*$", re.I),
     _ABOUT),
]


def small_talk(q):
    for pattern, reply in _SMALL_TALK:
        if pattern.match(q.strip()):
            return reply
    return None


def read_query(body):
    """Normalise the request. Unknown roles and modes fall back rather than
    erroring - the role also drives RBAC, so it must never be attacker-chosen
    into an unmapped value."""
    q = (body.get("q") or "").strip()
    role = body.get("role", "student")
    if role not in ("student", "faculty"):
        role = "student"
    mode = body.get("mode", "hybrid")
    if mode not in VALID_MODES:
        mode = "hybrid"
    return q, role, mode


def cite_payload(chunks):
    """Citation metadata plus a redacted preview, so a claim can be checked in
    the UI without opening the source PDF. redact() keeps emails and phone
    numbers out of the response exactly as it does out of the prompt."""
    from preprocessing.enrich import redact
    out = []
    for c in chunks:
        out.append({
            "url": c["url"], "page": c["page"], "category": c.get("category"),
            "score": c.get("score"), "published": c.get("published"),
            "department": c.get("department"),
            "regulation": c.get("regulation"),
            "academic_year": c.get("academic_year"),
            "snippet": redact(c["text"][:600]),
        })
    return out


@app.route("/api/status")
def status():
    r = get_retriever()
    path, kind = pick_model()
    return jsonify({"chunks": len(r.meta) if r else 0,
                    "dense": bool(r and r.dense is not None),
                    "ready": r is not None,
                    "model": os.path.basename(path) if path else None,
                    "kind": kind, "llama_server": llama_up(),
                    "refresh": refresher.status()})


@app.route("/api/refresh", methods=["POST"])
def refresh_now():
    """Manual trigger. The background loop already runs on its own; this is for
    'I just published a notice, pick it up now'."""
    force = bool((request.get_json(silent=True) or {}).get("force"))
    started = refresher.trigger(force=force)
    return jsonify({"started": started, "refresh": refresher.status()})


@app.route("/api/ask", methods=["POST"])
def ask():
    """Non-streaming answer. Kept alongside /api/ask/stream for scripted callers
    and for anything that cannot consume SSE."""
    q, role, mode = read_query(request.get_json(force=True))

    reply = small_talk(q)
    if reply:
        return jsonify({"answer": reply, "citations": [], "t_retrieve": 0.0,
                        "t_gen": 0.0, "mode": "conversation"})

    ret = get_retriever()
    if ret is None:
        st = refresher.status()
        return jsonify({"answer": "The knowledge base is still being built (%s). "
                                  "This runs automatically - try again shortly."
                                  % (st.get("detail") or st.get("phase")),
                        "citations": [], "t_retrieve": 0.0, "t_gen": 0.0, "mode": mode})

    t0 = time.time()
    chunks = ret.search(q, role=role, topk=5, mode=mode)
    t_ret = round(time.time() - t0, 2)

    if not chunks:
        return jsonify({"answer": ABSTAIN, "citations": [],
                        "t_retrieve": t_ret, "t_gen": 0.0, "mode": mode})

    msgs = build_prompt(q, chunks, role)
    t1 = time.time()
    with _lock:
        if llama_up():
            answer = llama_chat(msgs)
        else:
            answer = ("(llama-server is not running - retrieval only.)\n\n"
                      "Start it with runtime/start_llama.cmd\n\nMost relevant passage:\n\n"
                      + chunks[0]["text"][:700])
    t_gen = round(time.time() - t1, 2)

    # Refusing when the answer was retrieved is the worst failure we can make,
    # so a passage that plainly answers overrides the model's refusal (APP-003).
    if "could not find this" in answer.lower():
        answer = rescue(q, chunks, ret.bm25.idf) or answer

    # Citations are computed from the text, not taken from the model (APP-002).
    answer, used, unsupported = attribute(answer, chunks)

    # Nothing in the answer came from any retrieved passage. The model
    # sometimes returns only its trained-in hedge ("confirm which regulation
    # batch applies ... provisions differ across regulation versions"), which
    # answers nothing and is phrased differently every time, so it is caught
    # here by what it lacks - support - rather than by matching its wording.
    if not used and "could not find this" not in answer.lower():
        alt = rescue(q, chunks, ret.bm25.idf)
        if alt:
            answer, used, unsupported = attribute(alt, chunks)
    answer = with_supplement(q, answer, chunks, ret.bm25.idf)
    used = used_from(answer, len(chunks))
    return jsonify({"answer": answer, "citations": cite_payload(chunks),
                    "used": used, "unsupported": unsupported,
                    "t_retrieve": t_ret, "t_gen": t_gen, "mode": mode})


@app.route("/api/ask/stream", methods=["POST"])
def ask_stream():
    """Server-sent events: 'meta' (citations, retrieval time) once retrieval is
    done, then a 'token' per piece of generated text, then 'done' with timings.
    Citations are sent before the answer so a claim can be checked as it lands."""
    q, role, mode = read_query(request.get_json(force=True))

    def sse(event, obj):
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(obj))

    def gen():
        reply = small_talk(q)
        if reply:
            yield sse("meta", {"citations": [], "t_retrieve": 0.0, "mode": "conversation"})
            yield sse("token", {"t": reply})
            yield sse("done", {"t_gen": 0.0, "tok_s": None})
            return
        ret = get_retriever()
        if ret is None:
            st = refresher.status()
            yield sse("meta", {"citations": [], "t_retrieve": 0.0, "mode": mode})
            yield sse("token", {"t": "The knowledge base is still being built (%s). "
                                     "This runs automatically - try again shortly."
                                     % (st.get("detail") or st.get("phase"))})
            yield sse("done", {"t_gen": 0.0, "tok_s": None})
            return

        t0 = time.time()
        chunks = ret.search(q, role=role, topk=5, mode=mode)
        t_ret = round(time.time() - t0, 2)

        if not chunks:
            yield sse("meta", {"citations": [], "t_retrieve": t_ret, "mode": mode})
            yield sse("token", {"t": ABSTAIN})
            yield sse("done", {"t_gen": 0.0, "tok_s": None})
            return

        yield sse("meta", {"citations": cite_payload(chunks),
                           "t_retrieve": t_ret, "mode": mode})

        msgs = build_prompt(q, chunks, role)
        t1 = time.time()
        pieces = 0
        # One llama-server, one generation at a time - the lock is what keeps a
        # second request from interleaving tokens into this stream.
        with _lock:
            if not llama_up():
                yield sse("token", {"t": "(llama-server is not running - retrieval only.)"
                                         "\n\nStart it with runtime/start_llama.cmd"
                                         "\n\nMost relevant passage:\n\n"
                                         + chunks[0]["text"][:700]})
            else:
                try:
                    acc, sent = "", 0
                    for piece in llama_stream(msgs):
                        pieces += 1
                        acc += piece
                        # Hold back anything that is, or may become, the model's
                        # own "Source:" line - it is replaced by computed
                        # attribution at the end (APP-002).
                        safe = safe_prefix_len(acc)
                        if safe > sent:
                            yield sse("token", {"t": acc[sent:safe]})
                            sent = safe
                    if "could not find this" in acc.lower():
                        acc = rescue(q, chunks, ret.bm25.idf) or acc
                    final, used, unsupported = attribute(acc, chunks)
                    if not used and "could not find this" not in final.lower():
                        alt = rescue(q, chunks, ret.bm25.idf)
                        if alt:
                            final, used, unsupported = attribute(alt, chunks)
                    final = with_supplement(q, final, chunks, ret.bm25.idf)
                    used = used_from(final, len(chunks))
                    yield sse("final", {"text": final, "used": used,
                                        "unsupported": unsupported})
                except Exception as e:
                    yield sse("error", {"message": "generation failed: %s" % e})

        dt = time.time() - t1
        yield sse("done", {"t_gen": round(dt, 2),
                           "tok_s": round(pieces / dt, 1) if pieces and dt > 0 else None})

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    print("loading retriever ...", flush=True)
    get_retriever()
    p, k = pick_model()
    print("model:", os.path.basename(p) if p else "NONE", "|", k)
    print("llama-server:", "UP" if llama_up() else "DOWN (run runtime/start_llama.cmd)")

    # Start the pipeline. The UI is served immediately from the existing index;
    # the refresh runs behind it and hot-swaps the retriever when it finishes.
    # Skip in the reloader's parent process so it does not run twice in debug.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "false":
        print("auto-refresh: on (%d source(s) due now)"
              % refresher.status()["due_now"], flush=True)
        refresher.start()

    app.run(host="127.0.0.1", port=7860, threaded=True, use_reloader=False)
