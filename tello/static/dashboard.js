/* FireDrone dispatcher dashboard — read-only audience view.
 *
 * Subscribes to the same /ws/events stream the operator console does
 * and renders incident state, the live mission feed, and the agent's
 * evidence frames. We never send a control command from this page.
 */

const $ = (id) => document.getElementById(id);

const els = {
  status:     $("status-pill"),
  clock:      $("now-clock"),
  banner:     $("incident-banner"),
  bannerPill: $("incident-pill"),
  title:      $("incident-title"),
  summary:    $("incident-summary"),
  meta:       $("incident-meta"),

  video:        $("video"),
  videoOverlay: $("video-overlay"),
  videoTag:     $("video-tag"),
  hud: {
    battery:  $("hud-battery"),
    altitude: $("hud-altitude"),
    flight:   $("hud-flight"),
    audio:    $("hud-audio-band"),
  },

  mission: {
    pill:        $("mission-pill"),
    id:          $("mission-id"),
    trigger:     $("mission-trigger"),
    elapsed:     $("mission-elapsed"),
    evidenceCnt: $("mission-evidence-count"),
    step:        $("mission-step"),
    feed:        $("mission-feed"),
  },

  evidence: {
    gallery: $("evidence-gallery"),
    count:   $("evidence-count"),
  },
};

const state = {
  missionId:  null,
  startedAt:  null,
  evidence:   0,
  timer:      null,
};

// ------------------------ clock + helpers ------------------------ //

function tickClock() {
  const d = new Date();
  els.clock.textContent =
    String(d.getHours()).padStart(2, "0") + ":" +
    String(d.getMinutes()).padStart(2, "0") + ":" +
    String(d.getSeconds()).padStart(2, "0");
}
tickClock();
setInterval(tickClock, 1000);

function setStatus(stateKey, text) {
  els.status.dataset.state = stateKey;
  els.status.textContent   = text;
}

function setBanner(verdict, title, summary, meta) {
  els.banner.dataset.verdict     = verdict;
  els.bannerPill.dataset.verdict = verdict;
  els.bannerPill.textContent     = ({
    real_fire: "Real fire",
    false_alarm: "False alarm",
    running: "Mission active",
    idle: "Standby",
    unknown: "Inconclusive",
  })[verdict] || verdict;
  els.title.textContent   = title;
  els.summary.textContent = summary || "";
  els.meta.textContent    = meta || "—";
}

function setMissionStep(text) {
  els.mission.step.textContent = text;
}

function appendFeed(kind, body) {
  const li = document.createElement("li");
  li.className = `feed-entry feed-${kind}`;
  const k = document.createElement("span"); k.className = "feed-kind"; k.textContent = kind;
  const b = document.createElement("span"); b.className = "feed-body";
  if (typeof body === "string") b.textContent = body;
  else b.appendChild(body);
  li.appendChild(k); li.appendChild(b);
  els.mission.feed.appendChild(li);
  els.mission.feed.scrollTop = els.mission.feed.scrollHeight;
}

function clearFeed() {
  els.mission.feed.replaceChildren();
}

function clearEvidence() {
  els.evidence.gallery.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "evidence-empty";
  empty.textContent = "Evidence frames the agent captures will appear here in order.";
  els.evidence.gallery.appendChild(empty);
  state.evidence = 0;
  els.evidence.count.textContent = "0";
  els.mission.evidenceCnt.textContent = "0";
}

function startTimer() {
  stopTimer();
  state.timer = setInterval(() => {
    if (!state.startedAt) return;
    const sec = (Date.now() - state.startedAt) / 1000;
    els.mission.elapsed.textContent = `${sec.toFixed(1)} s`;
  }, 200);
}
function stopTimer() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }

// ------------------------ video ------------------------ //

els.video.onload  = () => { els.videoOverlay.classList.add("hidden"); els.videoTag.textContent = "live"; };
els.video.onerror = () => { els.videoOverlay.classList.remove("hidden"); els.videoTag.textContent = "offline"; };
els.video.src = `/video.mjpg?t=${Date.now()}`;

// ------------------------ telemetry ------------------------ //

function connectTelemetry() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  ws.onmessage = (ev) => {
    let p; try { p = JSON.parse(ev.data); } catch (_) { return; }
    const t = p.telemetry || {};
    setText(els.hud.battery,  t.battery_pct);
    setText(els.hud.altitude, t.tof_cm);
    setText(els.hud.flight,   t.flight_time_s);
  };
  ws.onclose = () => setTimeout(connectTelemetry, 1500);
  ws.onerror = () => ws.close();
}
function setText(el, v) {
  if (v === null || v === undefined) el.textContent = "—";
  else if (typeof v === "number" && !Number.isInteger(v)) el.textContent = v.toFixed(1);
  else el.textContent = v;
}

// ------------------------ events bus ------------------------ //

function connectEvents() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/events`);
  ws.onmessage = (ev) => {
    let p; try { p = JSON.parse(ev.data); } catch (_) { return; }
    const h = handlers[p.type];
    if (h) h(p);
  };
  ws.onclose = () => setTimeout(connectEvents, 1500);
  ws.onerror = () => ws.close();
}

const handlers = {};

handlers["audio_level"] = (p) => {
  if (typeof p.alarm_band_db === "number") {
    els.hud.audio.textContent = p.alarm_band_db.toFixed(0);
  }
};

handlers["audio_alarm"] = (p) => {
  if (p.state === "alarm") {
    appendFeed("alarm", `audio alarm (${p.source || "?"}): ${p.reason || "tone detected"}`);
  }
};

handlers["agent_state"] = (p) => {
  if (p.mission_id && p.mission_id !== state.missionId) {
    state.missionId = p.mission_id;
    state.startedAt = Date.now();
    state.evidence  = 0;
    clearFeed();
    clearEvidence();
    els.mission.id.textContent = p.mission_id;
    els.mission.trigger.textContent = p.trigger || "—";
    setBanner("running", "Mission active — inspecting the area",
      "The agent is autonomously sweeping the room. Verdict will appear here as soon as it submits a finding.",
      "MISSION " + p.mission_id);
    setStatus("running", "Mission active");
    setMissionStep("Spinning up");
    startTimer();
  }

  els.mission.pill.dataset.state = p.state;
  els.mission.pill.textContent   = p.state;
  appendFeed("state", `mission ${p.state}${p.error ? " — " + p.error : ""}`);

  if (p.state === "done" || p.state === "error") {
    stopTimer();
  }
};

handlers["agent_message"] = (p) => {
  setMissionStep(truncate(p.content || "", 220));
};

handlers["agent_tool_call"] = (p) => {
  setMissionStep(`Calling ${p.tool}(${shortArgs(p.args)})`);
  const span = document.createElement("span");
  const s = document.createElement("strong"); s.textContent = p.tool;
  span.appendChild(s);
  span.appendChild(document.createTextNode(`(${shortArgs(p.args)})`));
  appendFeed("tool", span);
};

handlers["agent_tool_result"] = (p) => {
  appendFeed("result", truncate(p.result || "", 200));
};

handlers["vision_result"] = (p) => {
  if (p.source !== "agent") return;
  // Drop the empty-state placeholder on first frame.
  const empty = els.evidence.gallery.querySelector(".evidence-empty");
  if (empty) empty.remove();
  els.evidence.gallery.appendChild(buildEvidenceTile(p));
  state.evidence += 1;
  els.evidence.count.textContent       = String(state.evidence);
  els.mission.evidenceCnt.textContent  = String(state.evidence);
  // Auto-scroll to newest frame.
  els.evidence.gallery.scrollLeft = els.evidence.gallery.scrollWidth;
};

handlers["incident"] = (p) => {
  stopTimer();
  const v = p.verdict || "unknown";
  const subtitleByVerdict = {
    real_fire:   "Fire confirmed. Fire department has been dispatched.",
    false_alarm: "No real fire detected — alarm logged with reasoning below.",
    unknown:     "Verdict inconclusive. Manual review recommended.",
  };
  setBanner(
    v,
    p.title || subtitleByVerdict[v] || "Incident",
    p.summary || subtitleByVerdict[v] || "",
    `${p.evidence?.length ?? 0} evidence · ${p.notified_dept ? "DISPATCHED" : "NO DISPATCH"} · ID ${p.incident_id}`,
  );
  setStatus(v, p.title || v);
  setMissionStep(p.summary || p.title || "Mission complete");

  // Reasoning list inside the feed.
  if (Array.isArray(p.reasons) && p.reasons.length) {
    const ul = document.createElement("div");
    ul.innerHTML = "<strong>verdict reasons:</strong>";
    const list = document.createElement("ul");
    list.style.margin = "4px 0 0 18px";
    list.style.padding = "0";
    for (const r of p.reasons) {
      const li = document.createElement("li");
      li.textContent = r;
      list.appendChild(li);
    }
    ul.appendChild(list);
    appendFeed("finding", ul);
  }
};

handlers["webhook_delivery"] = (p) => {
  appendFeed("state", p.error
    ? `webhook ${p.url} failed: ${p.error}`
    : `webhook ${p.url} -> ${p.status}`);
};

handlers["perception_alert"] = (p) => {
  appendFeed("state", `perception: ${p.reason || p.kind || "alert"} -> ${p.action || ""}`);
};

// ------------------------ helpers ------------------------ //

function buildEvidenceTile(p) {
  const tile = document.createElement("div");
  tile.className = "evidence-tile";
  tile.dataset.severity = p.severity || "none";

  if (p.thumbnail_b64) {
    const img = document.createElement("img");
    img.src = `data:image/jpeg;base64,${p.thumbnail_b64}`;
    img.alt = "evidence frame";
    tile.appendChild(img);
  }

  const body = document.createElement("div");
  body.className = "evidence-tile-body";

  const head = document.createElement("div");
  head.className = "evidence-tile-head";

  const sev = document.createElement("span");
  sev.className = "evidence-sev";
  sev.dataset.severity = p.severity || "none";
  sev.textContent = p.severity || "none";
  head.appendChild(sev);

  const flags = document.createElement("span");
  flags.className = "evidence-flags";
  if (p.fire_visible) {
    const f = document.createElement("span"); f.className = "evidence-flag on-red"; f.textContent = "fire"; flags.appendChild(f);
  }
  if (p.smoke_visible) {
    const s = document.createElement("span"); s.className = "evidence-flag on"; s.textContent = "smoke"; flags.appendChild(s);
  }
  head.appendChild(flags);

  body.appendChild(head);

  const desc = document.createElement("div");
  desc.className = "evidence-desc";
  desc.textContent = p.description || "";
  body.appendChild(desc);

  tile.appendChild(body);
  return tile;
}

function shortArgs(args) {
  if (!args) return "";
  try {
    const s = JSON.stringify(args);
    return s.length > 70 ? s.slice(0, 67) + "…" : s;
  } catch (_) { return ""; }
}

function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

// ------------------------ boot ------------------------ //

setBanner("idle", "Waiting for an alarm",
  "Drone is on standby. When the smoke alarm fires the agent will take off, sweep the room, and report its verdict here.",
  "—");
setStatus("idle", "Standby");
clearEvidence();

connectTelemetry();
connectEvents();

// Replay the last known incident so a refresh mid-event isn't blank.
(async function fetchLatest() {
  try {
    const res = await fetch("/api/incidents/latest");
    const data = await res.json();
    if (data.incident) handlers["incident"](data.incident);
  } catch (_) { /* offline ok */ }
})();
