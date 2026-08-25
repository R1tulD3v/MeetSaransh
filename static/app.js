"use strict";

// ------------------------------------------------------------------ tiny helpers
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Parse "mm:ss" or "h:mm:ss" -> seconds. Returns null if unparseable.
function tsToSeconds(ts) {
  if (!ts) return null;
  const parts = String(ts).split(":").map(Number);
  if (parts.some(isNaN)) return null;
  return parts.reduce((acc, n) => acc * 60 + n, 0);
}
function secondsToTs(sec) {
  sec = Math.floor(sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

let toastTimer;
function toast(msg, isError) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 3800);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// ------------------------------------------------------------------ state
let currentId = null;

// ------------------------------------------------------------------ health / badge
async function loadHealth() {
  try {
    const h = await api("/api/health");
    const badge = $("#status-badge");
    if (h.has_api_key) {
      badge.textContent = "Groq connected · " + h.asr_model;
      badge.className = "badge badge-ok";
    } else {
      badge.textContent = "No API key — sample mode";
      badge.className = "badge badge-warn";
      $("#upload-hint").textContent = "No GROQ_API_KEY set. Use “Load sample meeting”, or add a key in .env to upload your own audio.";
    }
  } catch (_) {
    $("#status-badge").textContent = "server offline";
  }
}

// ------------------------------------------------------------------ meeting list
async function loadMeetings() {
  const list = $("#meeting-list");
  const meetings = await api("/api/meetings");
  populateScope(meetings);
  list.innerHTML = "";
  if (!meetings.length) {
    list.appendChild(el("li", "empty", "No meetings yet."));
    return;
  }
  for (const m of meetings) {
    const li = el("li", "item");
    li.dataset.id = m.id;
    if (m.id === currentId) li.classList.add("active");
    const title = el("div", "mi-title", m.title);
    const meta = el("div", "mi-meta", `${new Date(m.created_at).toLocaleString()} · ${secondsToTs(m.duration)}`);
    li.append(title, meta);
    li.addEventListener("click", () => openMeeting(m.id));
    list.appendChild(li);
  }
}

function populateScope(meetings) {
  const sel = $("#chat-scope");
  const prev = sel.value;
  sel.innerHTML = '<option value="">All meetings</option>';
  for (const m of meetings) {
    const opt = el("option", null, m.title);
    opt.value = m.id;
    sel.appendChild(opt);
  }
  sel.value = prev;
}

// ------------------------------------------------------------------ open / render
async function openMeeting(id) {
  currentId = id;
  showLoader(false);
  const m = await api("/api/meetings/" + id);
  renderMeeting(m);
  document.querySelectorAll("#meeting-list .item").forEach((li) =>
    li.classList.toggle("active", li.dataset.id === id));
}

function seekTo(seconds) {
  const audio = $("#audio-player");
  if (audio.classList.contains("hidden") || isNaN(audio.duration)) {
    toast("No audio attached to this meeting (sample has no audio).");
    return;
  }
  audio.currentTime = seconds;
  audio.play().catch(() => {});
}

// Open a meeting from a citation and jump to the cited moment in the transcript.
async function openMeetingAt(meetingId, seconds) {
  switchView("meetings");
  await openMeeting(meetingId);
  switchTab("transcript");
  const rows = [...document.querySelectorAll("#transcript .seg")];
  // find the last segment whose start <= seconds
  let target = rows[0];
  for (const row of rows) {
    if (Number(row.dataset.start) <= seconds) target = row; else break;
  }
  if (target) {
    target.scrollIntoView({ block: "center" });
    target.classList.add("seg-flash");
    setTimeout(() => target.classList.remove("seg-flash"), 1600);
  }
  seekTo(seconds);
}

function tsChip(ts) {
  const sec = tsToSeconds(ts);
  if (sec == null) return null;
  const a = el("span", "ts", ts);
  a.addEventListener("click", () => seekTo(sec));
  return a;
}

function renderMeeting(m) {
  $("#empty-state").classList.add("hidden");
  $("#meeting-view").classList.remove("hidden");
  $("#mv-title").textContent = m.title;
  $("#mv-meta").textContent =
    `${new Date(m.created_at).toLocaleString()} · ${secondsToTs(m.duration)} · ${m.segments.length} segments`;

  // audio
  const audio = $("#audio-player");
  if (m.audio_ext) {
    audio.src = "/api/meetings/" + m.id + "/audio";
    audio.classList.remove("hidden");
  } else {
    audio.removeAttribute("src");
    audio.classList.add("hidden");
  }

  renderSummary(m.summary);
  renderTranscript(m.segments);
  switchTab("summary");
}

function renderSummary(s) {
  s = s || {};
  // TL;DR
  const tldr = $("#tldr");
  tldr.innerHTML = "";
  if (s.tldr) {
    tldr.append(el("h3", null, "TL;DR"), el("p", "tldr-text", s.tldr));
  }

  // Decisions
  const dec = $("#decisions");
  dec.innerHTML = "";
  if ((s.key_decisions || []).length) {
    dec.append(el("h3", null, "Key decisions"));
    const ul = el("ul", "clean");
    for (const d of s.key_decisions) {
      const li = el("li");
      li.append(document.createTextNode(d.decision));
      const chip = tsChip(d.timestamp);
      if (chip) li.append(chip);
      ul.appendChild(li);
    }
    dec.appendChild(ul);
  }

  // Action items table
  const act = $("#actions");
  act.innerHTML = "";
  if ((s.action_items || []).length) {
    act.append(el("h3", null, `Action items (${s.action_items.length})`));
    const table = el("table", "actions");
    const thead = el("thead");
    thead.innerHTML = "<tr><th>Task</th><th>Owner</th><th>Due</th><th></th></tr>";
    table.appendChild(thead);
    const tbody = el("tbody");
    for (const a of s.action_items) {
      const tr = el("tr");
      tr.appendChild(el("td", null, a.task));
      const ownerTd = el("td");
      const unassigned = /unassigned/i.test(a.owner || "");
      ownerTd.appendChild(el("span", "owner-chip" + (unassigned ? " owner-unassigned" : ""), a.owner || "Unassigned"));
      tr.appendChild(ownerTd);
      tr.appendChild(el("td", null, a.due || "Not specified"));
      const tsTd = el("td");
      const chip = tsChip(a.timestamp);
      if (chip) tsTd.appendChild(chip);
      tr.appendChild(tsTd);
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    act.appendChild(table);
  }

  // Open questions
  const q = $("#questions");
  q.innerHTML = "";
  if ((s.open_questions || []).length) {
    q.append(el("h3", null, "Open questions"));
    const ul = el("ul", "clean");
    for (const item of s.open_questions) ul.appendChild(el("li", null, item));
    q.appendChild(ul);
  }

  // Topics
  const top = $("#topics");
  top.innerHTML = "";
  if ((s.topics || []).length) {
    top.append(el("h3", null, "Topic timeline"));
    const ul = el("ul", "clean");
    for (const t of s.topics) {
      const li = el("li");
      const strong = el("strong", null, t.title);
      li.append(strong, document.createTextNode(" — " + t.summary));
      const chip = tsChip(t.timestamp);
      if (chip) li.append(chip);
      ul.appendChild(li);
    }
    top.appendChild(ul);
  }

  if (!s.tldr && !(s.key_decisions || []).length && !(s.action_items || []).length) {
    tldr.append(el("p", "meta", "No summary content was generated for this meeting."));
  }
}

function renderTranscript(segments) {
  const box = $("#transcript");
  box.innerHTML = "";
  if (!segments.length) {
    box.appendChild(el("p", "meta", "No transcript segments available."));
    return;
  }
  for (const seg of segments) {
    const row = el("div", "seg");
    row.dataset.start = seg.start;
    const ts = el("div", "seg-ts", secondsToTs(seg.start));
    ts.addEventListener("click", () => seekTo(seg.start));
    const text = el("div", "seg-text", seg.text);
    row.append(ts, text);
    box.appendChild(row);
  }
}

// ------------------------------------------------------------------ tabs + search
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("tab-active", t.dataset.tab === name));
  $("#tab-summary").classList.toggle("hidden", name !== "summary");
  $("#tab-transcript").classList.toggle("hidden", name !== "transcript");
}
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => switchTab(t.dataset.tab)));

$("#transcript-search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  document.querySelectorAll("#transcript .seg").forEach((row) => {
    const text = row.querySelector(".seg-text");
    const raw = text.textContent;
    const hit = !q || raw.toLowerCase().includes(q);
    row.classList.toggle("hidden-search", !hit);
    if (q && hit) {
      const i = raw.toLowerCase().indexOf(q);
      text.innerHTML = esc(raw.slice(0, i)) + "<mark>" + esc(raw.slice(i, i + q.length)) + "</mark>" + esc(raw.slice(i + q.length));
    } else {
      text.textContent = raw;
    }
  });
});

// ------------------------------------------------------------------ actions
function showLoader(on, text) {
  $("#loader").classList.toggle("hidden", !on);
  if (text) $("#loader-text").textContent = text;
  if (on) { $("#meeting-view").classList.add("hidden"); $("#empty-state").classList.add("hidden"); }
}

async function uploadMeeting() {
  const fileInput = $("#file-input");
  const file = fileInput.files[0];
  const hint = $("#upload-hint");
  hint.className = "hint";
  if (!file) { hint.textContent = "Choose an audio file first."; return; }

  const form = new FormData();
  form.append("file", file);
  form.append("title", $("#title-input").value.trim());

  $("#upload-btn").disabled = true;
  showLoader(true, "Transcribing & summarizing… this can take a moment.");
  try {
    const m = await api("/api/meetings", { method: "POST", body: form });
    currentId = m.id;
    await loadMeetings();
    showLoader(false);
    renderMeeting(m);
    document.querySelectorAll("#meeting-list .item").forEach((li) =>
      li.classList.toggle("active", li.dataset.id === m.id));
    fileInput.value = ""; $("#title-input").value = "";
    toast("Meeting processed ✓");
  } catch (err) {
    showLoader(false);
    hint.className = "hint error";
    hint.textContent = err.message;
    toast(err.message, true);
    if (!currentId) $("#empty-state").classList.remove("hidden");
  } finally {
    $("#upload-btn").disabled = false;
  }
}

async function loadSample() {
  $("#sample-btn").disabled = true;
  showLoader(true, "Loading sample meeting…");
  try {
    const m = await api("/api/meetings/sample", { method: "POST" });
    currentId = m.id;
    await loadMeetings();
    showLoader(false);
    renderMeeting(m);
    document.querySelectorAll("#meeting-list .item").forEach((li) =>
      li.classList.toggle("active", li.dataset.id === m.id));
    toast("Sample meeting loaded ✓");
  } catch (err) {
    showLoader(false);
    toast(err.message, true);
    if (!currentId) $("#empty-state").classList.remove("hidden");
  } finally {
    $("#sample-btn").disabled = false;
  }
}

async function exportMarkdown() {
  if (!currentId) return;
  try {
    const res = await fetch("/api/meetings/" + currentId + "/export");
    const md = await res.text();
    await navigator.clipboard.writeText(md);
    toast("Markdown copied to clipboard ✓");
  } catch (_) {
    toast("Could not copy to clipboard.", true);
  }
}

async function deleteMeeting() {
  if (!currentId) return;
  if (!confirm("Delete this meeting? This cannot be undone.")) return;
  const id = currentId;
  try {
    await api("/api/meetings/" + id, { method: "DELETE" });
    currentId = null;
    $("#meeting-view").classList.add("hidden");
    $("#empty-state").classList.remove("hidden");
    await loadMeetings();
    toast("Meeting deleted.");
  } catch (err) {
    toast(err.message, true);
  }
}

// ------------------------------------------------------------------ views + chat
function switchView(view) {
  document.querySelectorAll(".navbtn").forEach((b) =>
    b.classList.toggle("navbtn-active", b.dataset.view === view));
  $(".layout").classList.toggle("hidden", view !== "meetings");
  $("#chat-view").classList.toggle("hidden", view !== "ask");
  if (view === "ask") {
    loadRagStatus();
    $("#chat-q").focus();
  }
}
document.querySelectorAll(".navbtn").forEach((b) =>
  b.addEventListener("click", () => switchView(b.dataset.view)));

async function loadRagStatus() {
  try {
    const s = await api("/api/rag/status");
    const mode = s.embeddings_available ? "semantic + keyword" : "keyword-only";
    $("#rag-status").textContent =
      `${s.indexed_meetings} meeting(s) indexed · ${s.total_chunks} chunks · ${mode} search`;
  } catch (_) {}
}

function addMsg(cls, buildInner) {
  const box = $("#chat-messages");
  const hint = box.querySelector(".chat-hint");
  if (hint) hint.remove();
  const node = el("div", "msg " + cls);
  buildInner(node);
  box.appendChild(node);
  box.scrollTop = box.scrollHeight;
  return node;
}

function renderCitations(container, citations) {
  if (!citations || !citations.length) return;
  const wrap = el("div", "citations");
  wrap.appendChild(el("div", "cit-label", `Sources (${citations.length})`));
  for (const cit of citations) {
    const b = el("button", "cit");
    const head = el("div", "cit-head", `${cit.meeting_title} · ${cit.timestamp}`);
    const snip = el("div", "cit-snip", cit.snippet);
    b.append(head, snip);
    b.addEventListener("click", () => openMeetingAt(cit.meeting_id, cit.start));
    wrap.appendChild(b);
  }
  container.appendChild(wrap);
}

async function askQuestion(question) {
  question = (question || "").trim();
  if (!question) return;
  addMsg("msg-user", (n) => (n.textContent = question));
  $("#chat-q").value = "";
  $("#chat-send").disabled = true;
  const typing = addMsg("typing", (n) => (n.textContent = "Searching your meetings…"));
  try {
    const scope = $("#chat-scope").value || null;
    const r = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, meeting_id: scope }),
    });
    typing.remove();
    const refused = r.mode === "refused" || r.mode === "empty";
    addMsg("msg-bot" + (refused ? " refused" : ""), (n) => {
      if (r.note) n.appendChild(el("div", "msg-note", r.note));
      if (r.answer) n.appendChild(el("div", "msg-answer", r.answer));
      else if (r.mode === "retrieval_only") n.appendChild(el("div", "msg-answer", "Here are the most relevant excerpts:"));
      renderCitations(n, r.citations);
    });
    loadRagStatus();
  } catch (err) {
    typing.remove();
    addMsg("msg-bot refused", (n) => (n.textContent = err.message));
  } finally {
    $("#chat-send").disabled = false;
    $("#chat-q").focus();
  }
}

$("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  askQuestion($("#chat-q").value);
});
document.addEventListener("click", (e) => {
  if (e.target.classList.contains("example-q")) askQuestion(e.target.textContent);
});

// ------------------------------------------------------------------ wire up
$("#upload-btn").addEventListener("click", uploadMeeting);
$("#sample-btn").addEventListener("click", loadSample);
$("#export-btn").addEventListener("click", exportMarkdown);
$("#delete-btn").addEventListener("click", deleteMeeting);

loadHealth();
loadMeetings();
