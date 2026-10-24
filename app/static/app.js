"use strict";
/* VJLM chat UI (vanilla JS). Talks to the existing backend only:
   POST /api/ask/stream, GET /api/status, POST /api/refresh.
   There is no sign-in; the role comes from the sidebar. Nothing is stored about
   the conversation: it lives on the page and is gone on reload. */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* Escape first, then add formatting, so model output can never inject markup.
   [1] [2] become clickable refs; escaping never produces '[' or ']'. */
const md = (s) => esc(s)
  .replace(/```([\s\S]*?)```/g, "<pre>$1</pre>")
  .replace(/`([^`\n]+)`/g, "<code>$1</code>")
  .replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
  .replace(/^(?:[-*] .+(?:\n|$))+/gm,
    (m) => "<ul>" + m.trim().split("\n").map((l) => "<li>" + l.slice(2) + "</li>").join("") + "</ul>")
  .replace(/\[(\d{1,2})\]/g, '<span class="ref" data-n="$1">$1</span>');

/* The server appends two things to an answer: the other relevant sentences from
   the sources ("Also stated in the sources:" + bullets) and a plain "Sources:"
   line. The first is useful but long, so it folds away under the answer; the
   second repeats what the source cards below already show. */
const MORE = "Also stated in the sources:";
function answerHTML(t) {
  let head = String(t || "").replace(/\n\s*Sources?:[\s\S]*$/, "");
  let extra = "";
  const i = head.indexOf(MORE);
  if (i >= 0) { extra = head.slice(i + MORE.length).trim(); head = head.slice(0, i); }
  const n = (extra.match(/^[-*] /gm) || []).length;
  return md(head.trim()) + (extra
    ? '<details class="more"><summary>More from the sources' + (n ? " (" + n + ")" : "") + "</summary>" + md(extra) + "</details>"
    : "");
}

/* The exact sentence rag/retrieve.py tells the model to emit when the context
   does not contain the answer. Spotting it lets the UI show abstention as
   trained behaviour rather than as a failed answer. */
const ABSTAIN = "could not find this in the available vnrvjiet sources";

const SUG = {
  student: ["What is the attendance requirement?", "Which regulation applies to third year B.Tech?",
            "What are the placement statistics?", "What scholarships are available?",
            "How do I apply for hostel accommodation?", "What are my internal marks?"],
  faculty: ["Show the latest circulars", "What do the regulations say about internal marks?",
            "Show the academic calendar", "Which policy documents cover research?",
            "Where can I find faculty forms?", "What are the placement statistics?"],
};
const TOPICS = ["Syllabus", "Admissions", "Placements", "Hostel", "Circulars"];

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode / quota */ } },
};

let cur = null, ctl = null, busy = false, streamChat = null;
let role = store.get("vjlm:role", "student");
if (role !== "student" && role !== "faculty") role = "student";

function toast(t) {
  const d = document.createElement("div");
  d.className = "toast"; d.textContent = t; document.body.append(d);
  setTimeout(() => d.remove(), 3200);
}
const toBottom = () => { $("scroll").scrollTop = $("scroll").scrollHeight; };

/* ---------- sidebar ---------- */

function newChat() { if (busy && ctl) ctl.abort(); cur = null; render(); document.body.classList.remove("drawer"); $("q").focus(); }
$("new").onclick = newChat;

function toggleSide() { document.body.classList.toggle(innerWidth < 800 ? "drawer" : "collapsed"); }
$("toggle").onclick = toggleSide;
$("scrim").onclick = () => document.body.classList.remove("drawer");

$("theme").onclick = () => {
  const d = document.documentElement, n = d.dataset.theme === "dark" ? "light" : "dark";
  d.dataset.theme = n; store.set("vjlm-theme", n);
};
$("keys").onclick = () => $("kb").showModal();

$("role").onchange = () => {
  role = $("role").value; store.set("vjlm:role", role); document.body.dataset.role = role;
  if (!cur || !cur.msgs.length) render();            // suggestions depend on role
};
$("mode").onchange = () => store.set("vjlm:mode", $("mode").value);

/* ---------- messages ---------- */

function cardHTML(c, i, cid, used) {
  const name = (c.url || "").split("/").pop() || c.url || "source";
  const bits = [c.page != null && "page " + c.page, c.category, c.department, c.regulation,
                c.academic_year, c.published, c.score != null && "score " + c.score]
    .filter(Boolean).map(esc);
  return '<div class="cite' + (used ? "" : " un") + '" id="' + cid + "-" + (i + 1) + '"><span class="n">[' + (i + 1) + ']</span>' +
    '<a href="' + esc(c.url) + '" target="_blank" rel="noopener">' + esc(name) + '</a>' +
    '<div class="meta">' + bits.join(" · ") + "</div>" +
    (c.snippet ? "<details><summary>Show passage</summary><p>" + esc(c.snippet) + "</p></details>" : "") + "</div>";
}

function srcsHTML(m, cid) {
  const cites = m.cites || [];
  if (!cites.length) return "";
  const used = new Set(m.used || []);
  const one = (k, u) => cardHTML(cites[k], k, cid, u);
  const idx = cites.map((_, k) => k);
  let h;
  if (used.size) {
    const rest = idx.filter((k) => !used.has(k));
    h = "<h4>Sources used</h4>" + idx.filter((k) => used.has(k)).map((k) => one(k, true)).join("") +
        (rest.length ? "<h4>Also retrieved, not used in the answer</h4>" + rest.map((k) => one(k, false)).join("") : "");
  } else {
    h = "<h4>Sources</h4>" + idx.map((k) => one(k, true)).join("");
  }
  if (m.unsupported && m.unsupported.length) {
    h += '<div class="why">Not backed by any retrieved passage: ' + esc(m.unsupported.join(" ")) + "</div>";
  }
  return '<div class="srcs">' + h + "</div>";
}

function metricsText(meta) {
  if (!meta) return "";
  const p = [];
  if (meta.mode) p.push("retriever: " + meta.mode);
  if (meta.t_retrieve != null) p.push("retrieved in " + meta.t_retrieve + "s");
  if (meta.t_gen) p.push("generated in " + meta.t_gen + "s");
  if (meta.tok_s) p.push(meta.tok_s + " tok/s");
  return esc(p.join(" · "));
}

function msgHTML(m, i, last) {
  if (m.r === "u") return '<div class="msg u"><div class="bub">' + esc(m.t) + "</div></div>";
  const cid = "c" + cur.id + "-" + i;
  const abstained = m.done && (m.t || "").toLowerCase().includes(ABSTAIN);
  const dots = '<span class="dots"><i></i><i></i><i></i></span>';
  const body = abstained
    ? '<div class="tag">▲ No grounded answer in the sources</div>' + answerHTML(m.t) +
      '<div class="why">VJLM is trained to refuse rather than guess when the retrieved passages do not contain the answer.</div>'
    : (answerHTML(m.t) || (m.done ? "" : dots));
  return '<div class="msg a"><span class="av">V</span><div class="ab' + (abstained ? " abstain" : "") + '" data-cid="' + cid + '">' +
    '<div class="body">' + body + "</div>" +
    (m.err ? '<div class="err">' + esc(m.err) + "</div>" : "") +
    srcsHTML(m, cid) +
    (m.done ? '<div class="acts" data-i="' + i + '"><button data-a="copy" title="Copy">⧉</button>' +
      (last ? '<button data-a="redo" title="Regenerate">↻</button>' : "") +
      (metricsText(m.meta) ? '<span class="metrics">' + metricsText(m.meta) + "</span>" : "") + "</div>" : "") +
    "</div></div>";
}

function emptyHTML() {
  return '<div class="empty"><h1>Ask about VNRVJIET</h1>' +
    "<p>Answers are generated locally on CPU from the college's own public documents, with a source for every claim. " +
    "If the documents don't contain the answer, VJLM says so instead of guessing.</p>" +
    '<div class="chips">' + TOPICS.map((t) => '<button data-s="Tell me about ' + t.toLowerCase() + ' at VNRVJIET">' + t + "</button>").join("") +
    '</div><div class="cards">' + SUG[role].map((s) => '<button data-s="' + esc(s) + '">' + esc(s) + "</button>").join("") + "</div></div>";
}

function render() {
  $("msgs").innerHTML = cur && cur.msgs.length
    ? cur.msgs.map((m, i) => msgHTML(m, i, i === cur.msgs.length - 1)).join("")
    : emptyHTML();
  toBottom();
}

$("msgs").onclick = (e) => {
  const s = e.target.closest("[data-s]"); if (s) return ask(s.dataset.s);

  const ref = e.target.closest(".ref");
  if (ref) {
    const cid = ref.closest(".ab").dataset.cid, target = document.getElementById(cid + "-" + ref.dataset.n);
    if (target) {
      document.querySelectorAll(".cite.hl").forEach((x) => x.classList.remove("hl"));
      target.classList.add("hl");
      const d = target.querySelector("details"); if (d) d.open = true;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      setTimeout(() => target.classList.remove("hl"), 1800);
    }
    return;
  }

  const b = e.target.closest("[data-a]"); if (!b || !cur) return;
  const m = cur.msgs[b.parentNode.dataset.i], a = b.dataset.a;
  if (a === "copy") navigator.clipboard.writeText(m.t).then(() => toast("Copied"), () => toast("Copy failed"));
  else if (a === "redo") { const q = cur.msgs[cur.msgs.length - 2].t; cur.msgs.length -= 2; ask(q); }
};

/* ---------- asking (Server-Sent Events) ---------- */

function setBusy(b) {
  busy = b;
  $("send").textContent = b ? "■" : "↑";
  $("send").setAttribute("aria-label", b ? "Stop" : "Send");
}

async function ask(q) {
  q = (q || $("q").value).trim();
  if (!q || busy) return;
  if (!cur) {
    cur = { id: Date.now().toString(36), title: q.slice(0, 50), ts: Date.now(), msgs: [] };
  }
  const chat = cur, m = { r: "a", t: "", cites: [], used: [], unsupported: [], meta: {}, done: false };
  chat.msgs.push({ r: "u", t: q }, m); chat.ts = Date.now();
  $("q").value = ""; grow(); streamChat = chat; setBusy(true); render();

  ctl = new AbortController();
  let buf = "", aborted = false;
  try {
    const res = await fetch("/api/ask/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q, role, mode: $("mode").value }), signal: ctl.signal,
    });
    if (!res.ok || !res.body) throw new Error("Server error (" + res.status + ")");
    const rd = res.body.getReader(), dec = new TextDecoder();
    for (;;) {
      const { done, value } = await rd.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) { handle(buf.slice(0, k), m, chat); buf = buf.slice(k + 2); }
    }
  } catch (e) {
    if (e.name === "AbortError") aborted = true;
    else m.err = e.message || "Cannot reach the server.";
  }
  if (aborted) m.t += (m.t ? "\n\n" : "") + "[stopped]";
  m.done = true; streamChat = null; setBusy(false);
  if (cur === chat) render();
}

function handle(block, m, chat) {
  let ev = "", data = "";
  for (const l of block.split("\n")) {
    if (l.startsWith("event:")) ev = l.slice(6).trim();
    else if (l.startsWith("data:")) data += l.slice(5).trim();
  }
  let d; try { d = JSON.parse(data); } catch (e) { return; }
  if (ev === "meta") { m.cites = d.citations || []; m.meta.t_retrieve = d.t_retrieve; m.meta.mode = d.mode; }
  else if (ev === "token") m.t += d.t || "";
  else if (ev === "final") { m.t = d.text ?? m.t; m.used = d.used || []; m.unsupported = d.unsupported || []; }  // server-computed attribution replaces the streamed draft
  else if (ev === "done") { m.meta.t_gen = d.t_gen; m.meta.tok_s = d.tok_s; }
  else if (ev === "error") m.err = d.message;
  if (cur !== chat) return;                                  // user switched chats mid-stream
  const b = document.querySelector("#msgs .msg.a:last-child .body");
  if (b) { b.innerHTML = answerHTML(m.t) || '<span class="dots"><i></i><i></i><i></i></span>'; toBottom(); }
}

function grow() { const q = $("q"); q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 200) + "px"; }
$("q").oninput = grow;
$("q").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } };
$("send").onclick = () => (busy ? ctl.abort() : ask());

document.onkeydown = (e) => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); newChat(); }
  else if (e.ctrlKey && e.key.toLowerCase() === "b") { e.preventDefault(); toggleSide(); }
  else if (!typing && e.key === "/") { e.preventDefault(); $("q").focus(); }
  else if (!typing && e.key === "?") $("kb").showModal();
};

/* ---------- status + refresh ---------- */

const PHASE = {
  discovering: "Finding new documents…", fetching: "Checking sources…", extracting: "Reading documents…",
  enriching: "Tagging…", indexing: "Rebuilding index…", reloading: "Applying update…",
};
const ago = (t) => {
  const s = Date.now() / 1000 - t;
  return s < 90 ? "just now" : s < 5400 ? Math.round(s / 60) + "m ago" : s < 172800 ? Math.round(s / 3600) + "h ago" : Math.round(s / 86400) + "d ago";
};

async function poll() {
  let next = 15000;
  const el = $("status");
  try {
    const s = await (await fetch("/api/status")).json();

    const [t, c, tip] = !s.ready
      ? ["Knowledge base is still being built", "warn", "It is built automatically; or run: python rag/index.py"]
      : !s.llama_server
        ? ["AI server offline — showing best matching passages", "warn", "Start it with: runtime\\start_llama.cmd"]
        : [(s.chunks || 0).toLocaleString() + " passages · AI online", "ok", s.model || ""];
    el.textContent = t; el.className = "status " + c; el.title = tip;

    // Dense / Hybrid need the dense index; fall back only when the chosen one is unusable.
    const dense = !!s.dense, sel = $("mode");
    for (const o of sel.options) {
      if (o.value === "dense" || o.value === "hybrid") {
        o.disabled = !dense;
        const base = o.value === "dense" ? "Dense" : "Hybrid";
        o.textContent = dense ? base : base + " (no index)";
      }
    }
    if (sel.selectedOptions[0] && sel.selectedOptions[0].disabled) sel.value = "fusion";

    const r = s.refresh || {}, work = r.running && r.phase !== "idle", btn = $("sync");
    btn.className = "ghost" + (work ? " work" : "") + (r.phase === "error" ? " err" : "");
    btn.textContent = work ? (PHASE[r.phase] || "Refreshing…") : r.phase === "error" ? "Refresh failed — retry" : "Refresh documents";
    btn.title = r.phase === "error" ? (r.error || "refresh failed")
      : work ? "Refresh in progress" + (r.detail ? ": " + r.detail : "")
      : (r.due_now ? r.due_now + " source(s) due" : "Up to date") + (r.last_cycle ? " · checked " + ago(r.last_cycle) : "") + " — click to check now";
    if (work) next = 1500;
  } catch (e) {
    el.textContent = "Cannot reach the server"; el.className = "status warn";
  }
  setTimeout(poll, next);
}

$("sync").onclick = async () => {
  try {
    const r = await fetch("/api/refresh", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const d = await r.json().catch(() => ({}));
    toast(!r.ok ? "Refresh failed" : d.started ? "Refreshing documents in the background" : "A refresh is already running");
  } catch (e) { toast("Cannot reach the server"); }
};

/* ---------- start ---------- */

(function init() {
  $("role").value = role; document.body.dataset.role = role;
  const sm = store.get("vjlm:mode", "fusion");
  if ([...$("mode").options].some((o) => o.value === sm)) $("mode").value = sm;

  render(); poll(); $("q").focus();
})();
