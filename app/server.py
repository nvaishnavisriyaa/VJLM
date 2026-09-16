"""VJLM web app - local CPU inference, role-aware retrieval, cited answers.

Auth is a Phase-1 role simulation. The RBAC gate it feeds is real: retrieval
filters by visibility/audience BEFORE anything reaches the prompt builder.
"""
import json, os, sys, time, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, Response
from rag.retrieve import Retriever, build_prompt

MODEL_DIR = os.path.join(ROOT, "quantization", "builds")
app = Flask(__name__)

_ret = None
_llm = None
_lock = threading.Lock()


def pick_model():
    """Prefer the fine-tuned VJLM build; fall back to the base model."""
    for name in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []:
        if name.startswith("vjlm") and name.endswith(".gguf"):
            return os.path.join(MODEL_DIR, name), "fine-tuned"
    for name in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []:
        if name.endswith(".gguf"):
            return os.path.join(MODEL_DIR, name), "base"
    return None, None


def get_retriever():
    global _ret
    if _ret is None:
        _ret = Retriever()
    return _ret


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
</style></head><body>
<header>
  <h1>VJLM <span>Vignana Jyothi Language Model</span></h1>
  <span class="badge" id="mode">loading…</span>
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
fetch('/api/status').then(r=>r.json()).then(s=>{
  document.getElementById('mode').textContent =
    s.model ? (s.kind+' · '+s.chunks.toLocaleString()+' chunks') : 'retrieval only';
});
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
    return jsonify({"chunks": len(r.meta), "dense": r.dense is not None,
                    "model": os.path.basename(path) if path else None,
                    "kind": kind, "llama_server": llama_up()})


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True)
    q = (body.get("q") or "").strip()
    role = body.get("role", "student")
    if role not in ("student", "faculty"):
        role = "student"

    t0 = time.time()
    chunks = get_retriever().search(q, role=role, topk=5)
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
    app.run(host="127.0.0.1", port=7860, threaded=True)
