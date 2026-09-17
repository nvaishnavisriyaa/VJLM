"""VJLM web app - local CPU inference, role-aware retrieval, cited answers.

Auth is a Phase-1 role simulation. The RBAC gate it feeds is real: retrieval
filters by visibility/audience BEFORE anything reaches the prompt builder.
"""
import json, os, sys, time, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, Response
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


LLAMA_URL = os.environ.get("VJLM_LLAMA_URL", "http://127.0.0.1:8080")


def llama_chat(messages, max_tokens=420, temperature=0.2):
    """Talk to a local llama-server (official build, ~39x faster than the
    generic llama-cpp-python wheel on this CPU - see QUANT/RUNTIME bench).
    Local HTTP to 127.0.0.1 is not an external API."""
    import urllib.request, urllib.error
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature, "stream": False}).encode()
    req = urllib.request.Request(LLAMA_URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()


def llama_up():
    import urllib.request
    try:
        with urllib.request.urlopen(LLAMA_URL + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>VJLM - Vignana Jyothi Language Model</title>
<style>
:root{--bg:#0e1420;--card:#172131;--line:#26344a;--fg:#e8eef7;--mut:#93a4bd;--ac:#5b9cf8;--ac2:#3ddc97}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,Segoe UI,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:16px;flex-wrap:wrap}
h1{font-size:17px;margin:0;letter-spacing:.3px}
h1 span{color:var(--mut);font-weight:400}
.badge{font-size:11px;padding:3px 9px;border-radius:99px;background:#1d2b40;color:var(--ac2);border:1px solid var(--line)}
main{max-width:920px;margin:0 auto;padding:24px 16px 60px}
.row{display:flex;gap:10px;flex-wrap:wrap}
select,input,button{font:inherit;border-radius:9px;border:1px solid var(--line);background:var(--card);color:var(--fg);padding:11px 13px}
input{flex:1;min-width:240px}
button{background:var(--ac);border-color:var(--ac);color:#04101f;font-weight:600;cursor:pointer;padding:11px 20px}
button:disabled{opacity:.5;cursor:default}
.ans{margin-top:22px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;white-space:pre-wrap}
.cites{margin-top:16px}
.c{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--ac);border-radius:9px;padding:11px 14px;margin-bottom:9px;font-size:13px}
.c a{color:var(--ac);text-decoration:none;word-break:break-all}
.c .m{color:var(--mut);font-size:12px;margin-top:4px}
.ex{color:var(--mut);font-size:13px;margin-top:14px}
.ex b{color:var(--fg);font-weight:500;cursor:pointer;border-bottom:1px dotted var(--mut)}
.sp{color:var(--mut);font-size:13px;margin-top:18px}
.badge.live{cursor:pointer}
.badge.work{color:#ffc86b;border-color:#5a4520}
.badge.err{color:#ff9b9b;border-color:#5a2020}
.dot{display:inline-block;width:6px;height:6px;border-radius:99px;background:currentColor;
     margin-right:6px;vertical-align:middle}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.badge.work .dot{animation:pulse 1.1s ease-in-out infinite}
</style></head><body>
<header>
  <h1>VJLM <span>Vignana Jyothi Language Model</span></h1>
  <span class="badge" id="mode">loading…</span>
  <span class="badge live" id="sync" title="Click to refresh the knowledge base now">…</span>
  <span class="badge">local CPU · no external API</span>
</header>
<main>
  <div class="row">
    <select id="role"><option value="student">Student</option><option value="faculty">Faculty</option></select>
    <input id="q" placeholder="Ask about regulations, calendar, placements, departments…"
           onkeydown="if(event.key==='Enter')go()">
    <button id="b" onclick="go()">Ask</button>
  </div>
  <div class="ex">Try:
    <b onclick="set(this)">What is the attendance requirement?</b> ·
    <b onclick="set(this)">Which regulation applies to third year B.Tech?</b> ·
    <b onclick="set(this)">What are the placement statistics?</b> ·
    <b onclick="set(this)">What are my internal marks?</b>
  </div>
  <div id="out"></div>
</main>
<script>
// The refresh runs in the background from app start. Poll so the badge shows
// what it is doing rather than the pipeline being invisible.
const PHASE={idle:'knowledge base current',discovering:'finding new documents',
  fetching:'checking sources',extracting:'reading documents',enriching:'tagging',
  indexing:'rebuilding index',reloading:'applying update',
  error:'refresh failed',disabled:'auto-refresh off'};
function ago(t){if(!t)return'';const s=Date.now()/1000-t;
  if(s<90)return'just now';if(s<5400)return Math.round(s/60)+'m ago';
  if(s<172800)return Math.round(s/3600)+'h ago';return Math.round(s/86400)+'d ago';}
async function poll(){
  try{
    const s=await (await fetch('/api/status')).json();
    document.getElementById('mode').textContent =
      s.model ? (s.kind+' · '+s.chunks.toLocaleString()+' chunks') : 'retrieval only';
    const r=s.refresh||{}, b=document.getElementById('sync');
    const busy=r.running&&r.phase!=='idle', err=r.phase==='error';
    b.className='badge live'+(busy?' work':'')+(err?' err':'');
    let txt=PHASE[r.phase]||r.phase||'';
    if(busy&&r.detail)txt=r.detail;
    else if(r.phase==='idle')txt=(r.due_now?r.due_now+' source(s) due':'up to date')
      +(r.last_cycle?' · '+ago(r.last_cycle):'');
    b.innerHTML='<span class="dot"></span>'+txt;
    b.title=err?(r.error||'refresh failed'):'Click to refresh the knowledge base now';
    // poll faster while work is happening so the UI tracks it
    setTimeout(poll, busy?1500:15000);
  }catch(e){setTimeout(poll,15000);}
}
poll();
document.getElementById('sync').onclick=()=>{
  fetch('/api/refresh',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({})}).then(()=>setTimeout(poll,300));
};
function set(e){document.getElementById('q').value=e.textContent;go()}
async function go(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const role=document.getElementById('role').value;
  const b=document.getElementById('b'); b.disabled=true; b.textContent='…';
  document.getElementById('out').innerHTML='<div class="sp">Retrieving and generating on CPU…</div>';
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({q,role})});
    const d=await r.json();
    let h='<div class="ans">'+esc(d.answer)+'</div>';
    if(d.citations.length){
      h+='<div class="cites">';
      d.citations.forEach((c,i)=>{h+='<div class="c">['+(i+1)+'] <a href="'+c.url+'" target="_blank">'
        +c.url.split('/').pop()+'</a><div class="m">page '+c.page+' · '+c.category+' · score '+c.score+'</div></div>';});
      h+='</div>';
    }
    h+='<div class="sp">retrieved in '+d.t_retrieve+'s · generated in '+d.t_gen+'s</div>';
    document.getElementById('out').innerHTML=h;
  }catch(e){document.getElementById('out').innerHTML='<div class="ans">Error: '+e+'</div>';}
  b.disabled=false; b.textContent='Ask';
}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


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
    body = request.get_json(force=True)
    q = (body.get("q") or "").strip()
    role = body.get("role", "student")
    if role not in ("student", "faculty"):
        role = "student"

    ret = get_retriever()
    if ret is None:
        st = refresher.status()
        return jsonify({"answer": "The knowledge base is still being built (%s). "
                                  "This runs automatically - try again shortly."
                                  % (st.get("detail") or st.get("phase")),
                        "citations": [], "t_retrieve": 0.0, "t_gen": 0.0})

    t0 = time.time()
    chunks = ret.search(q, role=role, topk=5)
    t_ret = round(time.time() - t0, 2)

    if not chunks:
        return jsonify({"answer": "I could not find this in the available VNRVJIET sources.",
                        "citations": [], "t_retrieve": t_ret, "t_gen": 0.0})

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

    return jsonify({
        "answer": answer,
        "citations": [{"url": c["url"], "page": c["page"], "category": c["category"],
                       "score": c["score"]} for c in chunks],
        "t_retrieve": t_ret, "t_gen": t_gen,
    })


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
