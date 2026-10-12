"use strict";

const $ = (id) => document.getElementById(id);
const thread = $("thread");

/* The exact sentence rag/retrieve.py instructs the model to emit when the
   context does not contain the answer. Detecting it lets the UI present
   abstention as trained behaviour rather than as a failed answer. */
const ABSTAIN = "could not find this in the available vnrvjiet sources";

let streaming = null;   // AbortController while a generation is in flight
let denseReady = false;

/* ---------- theme ---------- */

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("vjlm-theme"); } catch (e) { /* private mode */ }
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  $("theme").onclick = () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const dark = cur === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", dark);
    try { localStorage.setItem("vjlm-theme", dark); } catch (e) { /* ignore */ }
  };
}

/* ---------- helpers ---------- */

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

/* Escape first, then turn [1] [2] into clickable refs. Escaping never emits
   '[' or ']', so this cannot be used to smuggle markup. */
function withRefs(s) {
  return esc(s).replace(/\[(\d{1,2})\]/g, '<span class="ref" data-n="$1">$1</span>');
}

function ago(t) {
  if (!t) return "";
  const s = Date.now() / 1000 - t;
  if (s < 90) return "just now";
  if (s < 5400) return Math.round(s / 60) + "m ago";
  if (s < 172800) return Math.round(s / 3600) + "h ago";
  return Math.round(s / 86400) + "d ago";
}

function scrollDown() {
  const m = $("main");
  m.scrollTop = m.scrollHeight;
}

/* ---------- status ---------- */

const PHASE = {
  idle: "knowledge base current", discovering: "finding new documents",
  fetching: "checking sources", extracting: "reading documents",
  enriching: "tagging", indexing: "rebuilding index",
  reloading: "applying update", error: "refresh failed",
  disabled: "auto-refresh off",
};

async function poll() {
  let next = 15000;
  try {
    const s = await (await fetch("/api/status")).json();

    $("model-txt").textContent = s.model
      ? s.kind + " · " + (s.chunks || 0).toLocaleString() + " chunks"
      : "retrieval only";
    $("chip-model").title = s.model ? s.model : "No GGUF build found";

    const up = !!s.llama_server;
    $("chip-llama").className = "chip " + (up ? "ok" : "warn");
    $("llama-txt").textContent = up ? "llama-server up" : "llama-server down";

    denseReady = !!s.dense;
    const modeSel = $("mode");
    for (const o of modeSel.options) {
      if (o.value === "dense" || o.value === "hybrid") {
        o.disabled = !denseReady;
        o.textContent = o.value === "dense"
          ? (denseReady ? "Dense" : "Dense (no index)")
          : (denseReady ? "Hybrid" : "Hybrid (no index)");
      }
    }
    // Only fall back when the chosen mode truly needs the dense index. This
    // used to reset coverage/lexical/hybrid3 to bm25 on every poll too.
    if (modeSel.selectedOptions[0] && modeSel.selectedOptions[0].disabled) modeSel.value = "bm25";

    const r = s.refresh || {};
    const busy = r.running && r.phase !== "idle";
    const err = r.phase === "error";
    $("chip-sync").className = "chip act" + (busy ? " work" : "") + (err ? " err" : "");
    let txt = PHASE[r.phase] || r.phase || "";
    if (busy && r.detail) txt = r.detail;
    else if (r.phase === "idle") {
      txt = (r.due_now ? r.due_now + " source(s) due" : "up to date")
          + (r.last_cycle ? " · " + ago(r.last_cycle) : "");
    }
    $("sync-txt").textContent = txt;
    $("chip-sync").title = err ? (r.error || "refresh failed")
                               : "Click to refresh the knowledge base now";

    notice(s);
    if (busy) next = 1500;
  } catch (e) {
    /* server not reachable - keep polling slowly */
  }
  setTimeout(poll, next);
}

function notice(s) {
  const slot = $("notice-slot");
  let html = "";
  if (!s.ready) {
    html = '<div class="notice"><b>Knowledge base not built yet.</b> '
         + 'The index is rebuildable and not in git — it is being built now, '
         + 'or run <code>python rag/index.py</code>.</div>';
  } else if (!s.llama_server) {
    html = '<div class="notice"><b>llama-server is not running</b> — answers will '
         + 'show the top retrieved passage instead of a generated answer. '
         + 'Start it with <code>runtime\\start_llama.cmd</code>.</div>';
  }
  if (slot.dataset.html !== html) { slot.innerHTML = html; slot.dataset.html = html; }
}

/* ---------- rendering ---------- */

function citeHTML(c, i) {
  const name = (c.url || "").split("/").pop() || c.url || "source";
  const bits = [];
  if (c.page != null) bits.push("page " + esc(c.page));
  if (c.category) bits.push(esc(c.category));
  if (c.department) bits.push(esc(c.department));
  if (c.regulation) bits.push(esc(c.regulation));
  if (c.academic_year) bits.push(esc(c.academic_year));
  if (c.score != null) bits.push("score " + esc(c.score));

  return '<div class="cite" id="cite-' + esc(c._turn) + "-" + (i + 1) + '">'
       +   '<div class="top"><span class="n">[' + (i + 1) + ']</span>'
       +     '<a href="' + esc(c.url) + '" target="_blank" rel="noopener">' + esc(name) + "</a>"
       +   "</div>"
       +   '<div class="meta">' + bits.map(b => "<span>" + b + "</span>").join("") + "</div>"
       +   (c.snippet
            ? '<button class="toggle">show passage</button>'
              + '<div class="snip">' + esc(c.snippet) + "</div>"
            : "")
       + "</div>";
}

function newTurn(question) {
  $("empty")?.remove();
  const id = "t" + Date.now();
  const el = document.createElement("div");
  el.className = "turn";
  el.id = id;
  el.innerHTML =
      '<div class="q"><span>' + esc(question) + "</span></div>"
    + '<div class="a" id="' + id + '-a">'
    +   '<div class="text" id="' + id + '-text"></div>'
    +   '<div id="' + id + '-cites"></div>'
    +   '<div class="metrics" id="' + id + '-m"><span>retrieving…</span></div>'
    + "</div>";
  thread.appendChild(el);
  scrollDown();
  return id;
}

function finishTurn(id, full, cites, meta) {
  const a = $(id + "-a");
  const textEl = $(id + "-text");
  const isAbstain = full.toLowerCase().includes(ABSTAIN);

  if (isAbstain) {
    a.classList.add("abstain");
    textEl.innerHTML =
        '<div class="tag">▲ No grounded answer in the sources</div>'
      + withRefs(full)
      + '<div class="why">VJLM is trained to refuse rather than guess when the '
      + 'retrieved passages do not contain the answer.</div>';
  } else {
    textEl.innerHTML = withRefs(full);
  }

  if (cites && cites.length) {
    // Show the passages that actually back the answer first; the rest are
    // retrieved context the answer did not use (APP-002).
    const used = new Set(meta.used || []);
    let html;
    if (used.size) {
      const u = cites.map((c, i) => used.has(i) ? citeHTML(c, i) : "").join("");
      const o = cites.map((c, i) => used.has(i) ? "" : citeHTML(c, i).replace('class="cite"', 'class="cite unused"')).join("");
      html = '<div class="cites"><h4>Sources used</h4>' + u + "</div>"
           + (o ? '<div class="cites"><h4>Also retrieved, not used in the answer</h4>' + o + "</div>" : "");
    } else {
      html = '<div class="cites"><h4>Sources</h4>' + cites.map(citeHTML).join("") + "</div>";
    }
    if (meta.unsupported && meta.unsupported.length) {
      html += '<div class="why">Not backed by any retrieved passage: ' + esc(meta.unsupported.join(" ")) + "</div>";
    }
    $(id + "-cites").innerHTML = html;
  }

  const m = [];
  if (meta.mode) m.push("retriever: " + esc(meta.mode));
  if (meta.t_retrieve != null) m.push("retrieved in " + esc(meta.t_retrieve) + "s");
  if (meta.t_gen != null) m.push("generated in " + esc(meta.t_gen) + "s");
  if (meta.tok_s) m.push(esc(meta.tok_s) + " tok/s");
  $(id + "-m").innerHTML = m.map(x => "<span>" + x + "</span>").join('<span class="sep">·</span>')
    + '<button class="copy">copy</button>';
  $(id + "-m").querySelector(".copy").onclick = (e) => {
    navigator.clipboard.writeText(full).then(() => {
      e.target.textContent = "copied";
      setTimeout(() => { e.target.textContent = "copy"; }, 1200);
    }).catch(() => { e.target.textContent = "failed"; });
  };
  scrollDown();
}

/* ---------- ask (streaming) ---------- */

async function ask() {
  const box = $("q");
  const q = box.value.trim();
  if (!q) return;

  if (streaming) { streaming.abort(); return; }

  box.value = "";
  box.style.height = "auto";
  const id = newTurn(q);
  const textEl = $(id + "-text");

  const btn = $("send");
  btn.textContent = "Stop";
  btn.classList.add("stop");

  streaming = new AbortController();
  let full = "", cites = [], meta = {};
  const cursor = '<span class="cursor"></span>';

  try {
    const res = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q, role: $("role").value, mode: $("mode").value }),
      signal: streaming.signal,
    });
    if (!res.ok || !res.body) throw new Error("HTTP " + res.status);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });

      // SSE frames are separated by a blank line
      let sep;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, sep);
        buf = buf.slice(sep + 2);

        let ev = "message", data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) ev = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;

        let d;
        try { d = JSON.parse(data); } catch (e) { continue; }

        if (ev === "meta") {
          cites = (d.citations || []).map(c => Object.assign({}, c, { _turn: id }));
          meta.t_retrieve = d.t_retrieve;
          meta.mode = d.mode;
          $(id + "-m").innerHTML = "<span>generating…</span>";
        } else if (ev === "token") {
          full += d.t || "";
          textEl.innerHTML = withRefs(full) + cursor;
          scrollDown();
        } else if (ev === "final") {
          // server-computed attribution replaces the streamed draft (APP-002)
          full = d.text || full;
          meta.used = d.used || [];
          meta.unsupported = d.unsupported || [];
          textEl.innerHTML = withRefs(full) + cursor;
        } else if (ev === "done") {
          meta.t_gen = d.t_gen;
          meta.tok_s = d.tok_s;
        } else if (ev === "error") {
          full += (full ? "\n\n" : "") + (d.message || "generation failed");
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") full += full ? "\n\n[stopped]" : "[stopped]";
    else full += (full ? "\n\n" : "") + "Error: " + e.message;
  }

  streaming = null;
  btn.textContent = "Ask";
  btn.classList.remove("stop");
  finishTurn(id, full, cites, meta);
  box.focus();
}

/* ---------- wiring ---------- */

document.addEventListener("click", (e) => {
  const t = $("q");

  const ex = e.target.closest("#examples button");
  if (ex) { t.value = ex.dataset.q; ask(); return; }

  const tog = e.target.closest(".cite .toggle");
  if (tog) {
    const c = tog.closest(".cite");
    c.classList.toggle("open");
    tog.textContent = c.classList.contains("open") ? "hide passage" : "show passage";
    return;
  }

  const ref = e.target.closest(".ref");
  if (ref) {
    const turn = ref.closest(".turn");
    const target = turn && turn.querySelector("#cite-" + turn.id + "-" + ref.dataset.n);
    if (target) {
      document.querySelectorAll(".cite.hl").forEach(x => x.classList.remove("hl"));
      target.classList.add("hl");
      if (!target.classList.contains("open")) {
        const b = target.querySelector(".toggle");
        if (b) { target.classList.add("open"); b.textContent = "hide passage"; }
      }
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      setTimeout(() => target.classList.remove("hl"), 1800);
    }
  }
});

$("send").onclick = ask;

$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
});

$("q").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
});

$("chip-sync").onclick = () => {
  fetch("/api/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  }).catch(() => {});
};

initTheme();
poll();
$("q").focus();
