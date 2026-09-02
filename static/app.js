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

// The API version lives here and nowhere else, so bumping it is a one-line change.
const API = "/api/v1";

// ------------------------------------------------------------------ session
// Tokens live in localStorage so a refresh does not sign the user out. That trades a
// little XSS exposure for usability; the mitigation is that the app has a strict CSP
// with no inline script, and the access token expires in 30 minutes.
const TOKEN_KEY = "meetsaransh.tokens";
let session = null;

function loadSession() {
  try {
    session = JSON.parse(localStorage.getItem(TOKEN_KEY) || "null");
  } catch (_) {
    session = null;
  }
  return session;
}

function saveSession(tokens) {
  session = tokens;
  try { localStorage.setItem(TOKEN_KEY, JSON.stringify(tokens)); } catch (_) {}
}

function clearSession() {
  session = null;
  try { localStorage.removeItem(TOKEN_KEY); } catch (_) {}
}

function authHeaders() {
  return session && session.access_token
    ? { Authorization: "Bearer " + session.access_token }
    : {};
}

// A single in-flight refresh, shared by every 401 that arrives while it runs. Without
// this, ten concurrent requests expiring together would fire ten refreshes and nine of
// them would fail, because refresh tokens are single-use.
let refreshInFlight = null;

async function refreshSession() {
  if (!session || !session.refresh_token) return false;
  if (!refreshInFlight) {
    refreshInFlight = fetch(API + "/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: session.refresh_token }),
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((tokens) => {
        if (tokens) { saveSession(tokens); return true; }
        clearSession();
        return false;
      })
      .catch(() => false)
      .finally(() => { refreshInFlight = null; });
  }
  return refreshInFlight;
}

async function api(path, opts, _retried) {
  opts = opts || {};
  const res = await fetch(API + path, {
    ...opts,
    headers: { ...(opts.headers || {}), ...authHeaders() },
  });

  // An expired access token is the normal case after 30 minutes, not an error: try one
  // silent refresh before bothering the user with a sign-in form.
  if (res.status === 401 && !_retried && session) {
    if (await refreshSession()) return api(path, opts, true);
    showAuthGate("Your session expired. Sign in again.");
    throw new Error("Session expired.");
  }

  if (!res.ok) {
    // The server returns {"error": {code, message, request_id}} on every failure.
    // `detail` is the older FastAPI shape, kept as a fallback.
    let message = res.statusText;
    let requestId = null;
    try {
      const body = await res.json();
      message = (body.error && body.error.message) || body.detail || message;
      requestId = body.error && body.error.request_id;
    } catch (_) {}
    const err = new Error(message);
    err.status = res.status;
    err.requestId = requestId;
    // A 429 is expected under load rather than broken, so it gets its own hint.
    if (res.status === 429) {
      const retry = res.headers.get("Retry-After");
      err.message = message + (retry ? ` (retry in ${retry}s)` : "");
    }
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

// ------------------------------------------------------------------ state
let currentId = null;
let pollTimer = null;
let authMode = "login";

// ------------------------------------------------------------------ auth UI
function showAuthGate(message) {
  clearSession();
  stopPolling();
  $("#auth-gate").classList.remove("hidden");
  const hint = $("#auth-hint");
  hint.className = message ? "hint error" : "hint";
  hint.textContent = message || "";
}

function hideAuthGate() {
  $("#auth-gate").classList.add("hidden");
  $("#auth-hint").textContent = "";
  $("#auth-password").value = "";
}

function setAuthMode(mode) {
  authMode = mode;
  const isLogin = mode === "login";
  $("#auth-tab-login").classList.toggle("auth-tab-active", isLogin);
  $("#auth-tab-register").classList.toggle("auth-tab-active", !isLogin);
  $("#auth-submit").textContent = isLogin ? "Sign in" : "Create account";
  $("#auth-password").setAttribute(
    "autocomplete", isLogin ? "current-password" : "new-password"
  );
  $("#auth-hint").textContent = "";
}

async function submitAuth(event) {
  event.preventDefault();
  const email = $("#auth-email").value.trim();
  const password = $("#auth-password").value;
  const hint = $("#auth-hint");
  const button = $("#auth-submit");

  hint.className = "hint";
  hint.textContent = authMode === "login" ? "Signing in…" : "Creating your account…";
  button.disabled = true;
  try {
    // Deliberately not via api(): there is no session yet, so the 401 refresh path
    // must not run -- a failed sign-in would otherwise recurse into itself.
    const res = await fetch(API + "/auth/" + (authMode === "login" ? "login" : "register"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error((body.error && body.error.message) || "Could not sign you in.");
    }
    saveSession(body);
    hideAuthGate();
    await startSession(body.user);
    toast(authMode === "login" ? "Signed in ✓" : "Account created ✓");
  } catch (err) {
    hint.className = "hint error";
    hint.textContent = err.message;
  } finally {
    button.disabled = false;
  }
}

async function signOut() {
  try { await api("/auth/logout", { method: "POST" }); } catch (_) {}
  currentId = null;
  $("#meeting-view").classList.add("hidden");
  $("#empty-state").classList.remove("hidden");
  $("#meeting-list").innerHTML = "";
  $("#user-email").textContent = "";
  showAuthGate("");
  setAuthMode("login");
}

async function startSession(user) {
  $("#user-email").textContent = user.email;
  await Promise.all([loadHealth(), loadMeetings()]);
  loadRagStatus();
}

/** Restore a session from localStorage, or show the gate. */
async function boot() {
  setAuthMode("login");
  if (!loadSession()) { showAuthGate(""); return; }
  try {
    const user = await api("/auth/me");
    hideAuthGate();
    await startSession(user);
  } catch (_) {
    showAuthGate("");
  }
}

// ------------------------------------------------------------------ health / badge
async function loadHealth() {
  try {
    const h = await api("/health");
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
  // Paginated response: {items, total, limit, offset}. The sidebar shows the most
  // recent page; `total` is what the count badge would use once paging lands in the UI.
  const page = await api("/meetings?limit=100");
  const meetings = page.items;
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
    // A meeting that is still processing is a real row, so it has to read as one that
    // is not finished rather than as one that is empty.
    if (m.status && m.status !== "done") {
      li.classList.add("pending");
      li.append(el("div", "mi-status " + m.status, STATUS_LABEL[m.status] || m.status));
    }
    li.addEventListener("click", () => openMeeting(m.id));
    list.appendChild(li);
  }
  syncPolling(meetings);
}

const STATUS_LABEL = {
  queued: "Queued…",
  processing: "Processing…",
  error: "Failed",
};

const STAGE_LABEL = {
  transcribing: "Transcribing the audio…",
  summarizing: "Writing the summary…",
  indexing: "Indexing for search…",
};

// ------------------------------------------------------------------ polling
// Polling runs only while something is actually in flight, and stops the moment the
// queue drains -- a timer that keeps firing forever is a battery and quota leak.
const POLL_MS = 2500;

function syncPolling(meetings) {
  const busy = meetings.some((m) => m.status === "queued" || m.status === "processing");
  if (busy && !pollTimer) {
    pollTimer = setInterval(pollOnce, POLL_MS);
  } else if (!busy) {
    stopPolling();
  }
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollOnce() {
  try {
    const before = currentId ? await api("/meetings/" + currentId).catch(() => null) : null;
    await loadMeetings();
    // Re-render the open meeting when it finishes, so the user does not have to click
    // away and back to see the summary appear.
    if (before && currentId === before.id) {
      if (before.status === "done") { renderMeeting(before); loadRagStatus(); }
      else renderProcessing(before);
    }
  } catch (_) {
    stopPolling();  // the session is gone or the server is down; stop hammering it
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
  const m = await api("/meetings/" + id);
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

/** Show or hide the progress panel, hiding the tabs while it is up. */
function setProcessingView(on) {
  $("#processing-panel").classList.toggle("hidden", !on);
  $("#detail-tabs").classList.toggle("hidden", on);
  $("#export-btn").disabled = on;  // there is no summary to copy yet
  if (on) {
    $("#tab-summary").classList.add("hidden");
    $("#tab-transcript").classList.add("hidden");
  }
  // Turning it off does not reveal a panel here: renderMeeting() calls switchTab()
  // right afterwards, which is the single place that decides which tab is showing.
}

/** A meeting that has no transcript yet gets a progress screen, not an empty one. */
function renderProcessing(m) {
  $("#empty-state").classList.add("hidden");
  $("#meeting-view").classList.remove("hidden");
  $("#mv-title").textContent = m.title;
  $("#mv-meta").textContent = new Date(m.created_at).toLocaleString();
  $("#audio-player").classList.add("hidden");

  const panel = $("#processing-panel");
  panel.innerHTML = "";

  if (m.status === "error") {
    panel.append(
      el("h2", null, "Processing failed"),
      el("p", null, m.error || "Something went wrong while processing this recording.")
    );
    const retry = el("button", "btn btn-ghost", "Delete and try again");
    retry.addEventListener("click", deleteMeeting);
    panel.append(retry);
  } else {
    panel.append(
      el("div", "spinner-ring"),
      el("h2", null, STATUS_LABEL[m.status] || "Working…"),
      el(
        "p",
        null,
        (STAGE_LABEL[m.stage] || "Your recording is in the queue.") +
          " You can close this tab - processing continues on the server."
      )
    );
  }
  setProcessingView(true);
}

function renderMeeting(m) {
  if (m.status && m.status !== "done") { renderProcessing(m); return; }
  setProcessingView(false);
  $("#empty-state").classList.add("hidden");
  $("#meeting-view").classList.remove("hidden");
  $("#mv-title").textContent = m.title;
  $("#mv-meta").textContent =
    `${new Date(m.created_at).toLocaleString()} · ${secondsToTs(m.duration)} · ${m.segments.length} segments`;

  // audio
  const audio = $("#audio-player");
  if (m.audio_ext) {
    audio.src = API + "/meetings/" + m.id + "/audio";
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
  showLoader(true, "Uploading…");
  try {
    // 202: the server has the file and has queued the work. From here the meeting is
    // polled rather than awaited, so a long recording cannot be lost to a timeout.
    const accepted = await api("/meetings", { method: "POST", body: form });
    currentId = accepted.id;
    await loadMeetings();
    showLoader(false);
    renderProcessing({ ...accepted, stage: null });
    document.querySelectorAll("#meeting-list .item").forEach((li) =>
      li.classList.toggle("active", li.dataset.id === accepted.id));
    fileInput.value = ""; $("#title-input").value = "";
    toast("Uploaded - transcribing in the background ✓");
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
    const m = await api("/meetings/sample", { method: "POST" });
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
    const res = await fetch(API + "/meetings/" + currentId + "/export");
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
    await api("/meetings/" + id, { method: "DELETE" });
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
  $("#view-insights").classList.toggle("hidden", view !== "insights");
  if (view === "ask") {
    loadRagStatus();
    $("#chat-q").focus();
  }
  // Fetched on entry rather than kept live: the numbers only change when a meeting
  // finishes, and polling aggregates nobody is looking at is wasted budget.
  if (view === "insights") loadInsights();
}
document.querySelectorAll(".navbtn").forEach((b) =>
  b.addEventListener("click", () => switchView(b.dataset.view)));

async function loadRagStatus() {
  try {
    const s = await api("/rag/status");
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
    const r = await api("/chat", {
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

// ------------------------------------------------------------------ insights
// Charts are inline SVG built with element attributes, not a charting library: it keeps
// the no-build-step story intact, and setting geometry via attributes rather than style
// works under the app's strict CSP with no exception.
const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  return node;
}

function statTile(label, value, sub) {
  const box = el("div", "stat");
  box.append(el("div", "stat-label", label), el("div", "stat-value", value));
  if (sub) box.appendChild(el("div", "stat-sub", sub));
  return box;
}

function panel(title, note) {
  const box = el("section", "panel");
  box.appendChild(el("h3", null, title));
  if (note) box.appendChild(el("p", "panel-note", note));
  return box;
}

function hoursLabel(seconds) {
  const mins = Math.round((seconds || 0) / 60);
  if (mins < 60) return `${mins} min`;
  return `${(mins / 60).toFixed(1)} hrs`;
}

/** Horizontal bars. The green portion is the share that has a real due date. */
function renderOwnerBars(rows) {
  const box = panel(
    "Action items by owner",
    "Green marks the share with a due date. Unassigned work is listed, not hidden."
  );
  if (!rows.length) {
    box.appendChild(el("p", "panel-empty", "No action items yet."));
    return box;
  }
  const max = Math.max(...rows.map((r) => r.total));
  const bars = el("div", "bars");
  for (const row of rows) {
    const line = el("div", "bar-row");
    line.appendChild(el("div", "bar-name", row.owner));

    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill" + (row.with_due_date ? " partial" : ""));
    // setProperty on a CSSOM object is not an inline style attribute, so the CSP
    // that forbids unsafe-inline styles is unaffected.
    fill.style.setProperty("width", `${(row.total / max) * 100}%`);
    track.appendChild(fill);
    line.appendChild(track);

    line.appendChild(
      el("div", "bar-count", `${row.total}${row.with_due_date ? ` (${row.with_due_date} dated)` : ""}`)
    );
    bars.appendChild(line);
  }
  box.appendChild(bars);
  return box;
}

/** Area-and-line sparkline over the window, with the last point emphasised. */
function renderTimeSeries(points, windowDays) {
  const box = panel(`Meetings over the last ${windowDays} days`, null);
  const total = points.reduce((sum, p) => sum + p.meetings, 0);
  if (!total) {
    box.appendChild(el("p", "panel-empty", "No meetings in this window."));
    return box;
  }

  const W = 640, H = 96, PAD = 6;
  const max = Math.max(1, ...points.map((p) => p.meetings));
  const stepX = (W - PAD * 2) / Math.max(1, points.length - 1);
  const y = (v) => H - PAD - (v / max) * (H - PAD * 2);
  const x = (i) => PAD + i * stepX;

  const chart = svg("svg", {
    class: "spark",
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: "none",
    role: "img",
    "aria-label": `${total} meetings over the last ${windowDays} days`,
  });

  chart.appendChild(svg("line", { class: "spark-grid", x1: PAD, y1: y(0), x2: W - PAD, y2: y(0) }));

  const line = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.meetings).toFixed(1)}`);
  chart.appendChild(
    svg("path", {
      class: "spark-area",
      d: `${line.join(" ")} L${x(points.length - 1).toFixed(1)},${y(0)} L${x(0).toFixed(1)},${y(0)} Z`,
    })
  );
  chart.appendChild(svg("path", { class: "spark-line", d: line.join(" ") }));

  const last = points[points.length - 1];
  chart.appendChild(
    svg("circle", { class: "spark-dot", cx: x(points.length - 1), cy: y(last.meetings), r: 3 })
  );

  box.appendChild(chart);
  const ends = el("div", "bar-row");
  ends.style.setProperty("grid-template-columns", "1fr auto");
  ends.append(
    el("div", "bar-count", new Date(points[0].day).toLocaleDateString()),
    el("div", "bar-count", `${total} meeting${total === 1 ? "" : "s"} · today`)
  );
  box.appendChild(ends);
  return box;
}

function renderTopics(rows) {
  const box = panel("Recurring topics", "Counted across every meeting, case-insensitively.");
  if (!rows.length) {
    box.appendChild(el("p", "panel-empty", "No topics yet."));
    return box;
  }
  const list = el("div", "topic-list");
  for (const row of rows) {
    const chip = el("div", "topic-chip");
    chip.appendChild(el("b", null, row.title));
    chip.appendChild(el("span", null, `${row.mentions}`));
    list.appendChild(chip);
  }
  box.appendChild(list);
  return box;
}

function renderLooseEnds(rows) {
  const box = panel("Work with no owner", "The loose ends worth chasing first.");
  if (!rows.length) {
    box.appendChild(el("p", "panel-empty", "Everything has an owner."));
    return box;
  }
  const list = el("ul", "loose-list");
  for (const row of rows) {
    const item = el("li");
    const button = el("button", "loose-item");
    button.type = "button";
    button.append(
      el("div", "loose-task", row.task),
      el("div", "loose-meta", `${row.meeting_title}${row.timestamp ? ` · ${row.timestamp}` : ""}`)
    );
    button.addEventListener("click", () => {
      switchView("meetings");
      openMeeting(row.meeting_id);
    });
    item.appendChild(button);
    list.appendChild(item);
  }
  box.appendChild(list);
  return box;
}

async function loadInsights() {
  const body = $("#insights-body");
  body.innerHTML = "";
  body.appendChild(el("p", "panel-empty", "Loading…"));
  try {
    const days = $("#insights-window").value;
    const data = await api(`/analytics?days=${days}`);
    body.innerHTML = "";

    const o = data.overview;
    const stats = el("div", "stat-row");
    stats.append(
      statTile("Meetings", String(o.meetings), hoursLabel(o.total_seconds) + " captured"),
      statTile("Action items", String(o.action_items)),
      statTile("Decisions", String(o.decisions)),
      statTile("Open questions", String(o.open_questions)),
      statTile("Unowned", String(data.unassigned.length), "need an owner")
    );
    body.appendChild(stats);

    body.appendChild(renderTimeSeries(data.over_time, data.window_days));

    const grid = el("div", "panel-grid");
    grid.append(renderOwnerBars(data.by_owner), renderLooseEnds(data.unassigned));
    body.appendChild(grid);

    body.appendChild(renderTopics(data.top_topics));
  } catch (err) {
    body.innerHTML = "";
    body.appendChild(el("p", "panel-empty", err.message));
  }
}

// ------------------------------------------------------------------ wire up
$("#upload-btn").addEventListener("click", uploadMeeting);
$("#sample-btn").addEventListener("click", loadSample);
$("#export-btn").addEventListener("click", exportMarkdown);
$("#delete-btn").addEventListener("click", deleteMeeting);
$("#auth-form").addEventListener("submit", submitAuth);
$("#auth-tab-login").addEventListener("click", () => setAuthMode("login"));
$("#auth-tab-register").addEventListener("click", () => setAuthMode("register"));
$("#logout-btn").addEventListener("click", signOut);
$("#insights-window").addEventListener("change", loadInsights);
// Leaving an interval running against a hidden tab wastes the user's battery and
// their rate-limit budget for no benefit.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopPolling();
  else loadMeetings().catch(() => {});
});

boot();
