/* FireDrone dispatcher dashboard — read-only audience view.
 *
 * Subscribes to the same /ws/events + /ws/telemetry streams the operator
 * console does and renders incident state, the live mission feed, and the
 * agent's evidence frames. We never send a control command from this page.
 *
 * New in the redesign:
 *   - SVG ring meters for battery + flight-time-vs-90s-cap; vertical bar
 *     for altitude; horizontal bar with threshold marker for 3 kHz audio
 *   - Four-phase mission stepper derived from agent_state + tool calls
 *   - Evidence tiles gain a severity stripe + relative timestamp
 *   - Verdict block (colored, reasoning bullets) is prepended to the feed
 *     when an incident fires; banner uses green/amber/red flat fills
 */

const $ = (id) => document.getElementById(id);

const els = {
  status:     $("status-pill"),
  clock:      $("now-clock"),
  banner:     $("incident-banner"),
  bannerPill: $("incident-pill"),
  title:      $("incident-title"),
  summary:    $("incident-summary"),
  dispatch:   $("incident-dispatch"),
  meta:       $("incident-meta"),

  video:        $("video"),
  videoOverlay: $("video-overlay"),
  videoTag:     $("video-tag"),

  hud: {
    battery:  $("hud-battery"),
    altitude: $("hud-altitude"),
    flight:   $("hud-flight"),
    audio:    $("hud-audio-band"),
    audioBox: $("hud-audio"),
    audioFill: $("audio-fill"),
    audioThreshold: $("audio-threshold"),
    altFill:  $("alt-fill"),
    batteryRing: $("battery-ring"),
    batteryPct:  $("battery-pct"),
    flightRing:  $("flight-ring"),
    flightPct:   $("flight-pct"),
  },

  mission: {
    pill:        $("mission-pill"),
    id:          $("mission-id"),
    trigger:     $("mission-trigger"),
    elapsed:     $("mission-elapsed"),
    evidenceCnt: $("mission-evidence-count"),
    step:        $("mission-step"),
    feed:        $("mission-feed"),
    stepper:     $("mission-stepper"),
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
  verdict:    null,        // most recent incident verdict
  audioLastDb: 0,
};

// ------------------------ constants ------------------------ //

/* 2 * pi * r where r=15.5 in the SVG (matches HTML markup). */
const RING_CIRC = 2 * Math.PI * 15.5;

/* Mission cap from agent.py — keep in sync if changed there. */
const MISSION_CAP_S = 90.0;

/* Phases used by the stepper. */
const PHASES = ["trigger", "inspect", "decide", "report"];

/* Audio band thresholds (dB). Anything above ALARM_DB lights the bar
   amber and is visually flagged as "in alarm". The bar is normalised
   between FLOOR_DB and CEIL_DB. */
const AUDIO_FLOOR_DB = -90;
const AUDIO_CEIL_DB  = -20;
const AUDIO_ALARM_DB = -50;

// position the threshold marker on the audio bar
(function placeAudioThreshold() {
  if (!els.hud.audioThreshold) return;
  const t = Math.max(0, Math.min(1,
    (AUDIO_ALARM_DB - AUDIO_FLOOR_DB) / (AUDIO_CEIL_DB - AUDIO_FLOOR_DB)));
  els.hud.audioThreshold.style.left = `${(t * 100).toFixed(1)}%`;
})();

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
  state.verdict = verdict;
  els.banner.dataset.verdict     = verdict;
  els.bannerPill.dataset.verdict = verdict;
  els.bannerPill.textContent     = ({
    real_fire:   "Real fire",
    false_alarm: "False alarm",
    running:     "Mission active",
    idle:        "Standby",
    unknown:     "Inconclusive",
  })[verdict] || verdict;
  els.title.textContent   = title;
  els.summary.textContent = summary || "";
  els.meta.textContent    = meta || "—";
}

function setDispatchBadge(text, on) {
  if (!els.dispatch) return;
  els.dispatch.textContent = text;
  els.dispatch.dataset.on = on ? "true" : "false";
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
  els.evidence.count.textContent       = "0";
  els.mission.evidenceCnt.textContent  = "0";
}

function startTimer() {
  stopTimer();
  state.timer = setInterval(() => {
    if (!state.startedAt) return;
    const sec = (Date.now() - state.startedAt) / 1000;
    els.mission.elapsed.textContent = sec.toFixed(1);
    setFlightRing(sec);
  }, 200);
}
function stopTimer() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }

// ------------------------ stepper ------------------------ //

function setStepStatus(phase, status) {
  if (!els.mission.stepper) return;
  const li = els.mission.stepper.querySelector(`.step[data-phase="${phase}"]`);
  if (li) li.dataset.status = status;
}

function resetStepper() {
  for (const p of PHASES) setStepStatus(p, "default");
}

/**
 * Advance the stepper to a named phase. Earlier phases get marked "done";
 * the named phase becomes "active". Past phases remain "done" on later
 * calls so the stepper never regresses.
 */
function advanceStepper(phase) {
  const idx = PHASES.indexOf(phase);
  if (idx < 0) return;
  PHASES.forEach((p, i) => {
    if (i < idx)       setStepStatus(p, "done");
    else if (i === idx) setStepStatus(p, "active");
    else                setStepStatus(p, "default");
  });
}

/**
 * Final stepper state for a verdict. All earlier phases are "done";
 * the "report" phase gets a verdict-coloured terminal status:
 * good (false alarm), bad (real fire), active (unknown/inconclusive).
 */
function markStepperVerdict(verdict) {
  for (const p of ["trigger", "inspect", "decide"]) setStepStatus(p, "done");
  const finalStatus = ({
    false_alarm: "good",
    real_fire:   "bad",
    unknown:     "active",
  })[verdict] || "done";
  setStepStatus("report", finalStatus);
}

// ------------------------ ring meters ------------------------ //

function setRing(ringEl, pct, state) {
  if (!ringEl) return;
  const clamped = Math.max(0, Math.min(100, Number.isFinite(pct) ? pct : 0));
  const fillEl = ringEl.querySelector(".ring-fill");
  if (fillEl) {
    const offset = RING_CIRC * (1 - clamped / 100);
    fillEl.setAttribute("stroke-dasharray", RING_CIRC.toFixed(2));
    fillEl.setAttribute("stroke-dashoffset", offset.toFixed(2));
  }
  ringEl.classList.remove("warn", "bad", "good");
  if (state) ringEl.classList.add(state);
}

function setBatteryRing(pct) {
  let s = null;
  if (typeof pct === "number") {
    if (pct < 15)      s = "bad";
    else if (pct < 30) s = "warn";
  }
  setRing(els.hud.batteryRing, pct, s);
  if (els.hud.batteryPct) {
    els.hud.batteryPct.textContent =
      (typeof pct === "number" && Number.isFinite(pct)) ? Math.round(pct) : "—";
  }
}

function setFlightRing(seconds) {
  const sec = Number.isFinite(seconds) ? seconds : 0;
  const pct = Math.max(0, Math.min(100, (sec / MISSION_CAP_S) * 100));
  let s = "good";
  if (pct >= 90)      s = "bad";
  else if (pct >= 70) s = "warn";
  setRing(els.hud.flightRing, pct, s);
  if (els.hud.flightPct) {
    els.hud.flightPct.textContent = `${Math.round(MISSION_CAP_S - sec)}`;
  }
}

function setAltBar(altCm) {
  if (!els.hud.altFill) return;
  // 0–300 cm range — Tello hover-and-rotate stays comfortably below that.
  const pct = Math.max(0, Math.min(100, (Number(altCm) || 0) / 300 * 100));
  els.hud.altFill.style.height = `${pct.toFixed(1)}%`;
}

// ------------------------ audio band bar ------------------------ //

function setAudioBar(db) {
  if (!els.hud.audioFill) return;
  const v = Number.isFinite(db) ? db : AUDIO_FLOOR_DB;
  const pct = Math.max(0, Math.min(100,
    (v - AUDIO_FLOOR_DB) / (AUDIO_CEIL_DB - AUDIO_FLOOR_DB) * 100));
  els.hud.audioFill.style.width = `${pct.toFixed(1)}%`;
  if (els.hud.audioBox) {
    els.hud.audioBox.classList.toggle("alarm", v >= AUDIO_ALARM_DB);
  }
  state.audioLastDb = v;
}

// ------------------------ video ------------------------ //

els.video.onload  = () => { els.videoOverlay.classList.add("hidden"); els.videoTag.textContent = "live"; els.videoTag.dataset.state = "live"; };
els.video.onerror = () => { els.videoOverlay.classList.remove("hidden"); els.videoTag.textContent = "offline"; els.videoTag.dataset.state = ""; };
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
    // Flight-time is driven by our own mission timer while a mission is
    // running so it advances smoothly between telemetry frames; fall
    // back to the server value when idle.
    if (state.startedAt === null) {
      setText(els.hud.flight, t.flight_time_s);
    }
    setBatteryRing(t.battery_pct);
    setAltBar(t.tof_cm);
  };
  ws.onclose = () => setTimeout(connectTelemetry, 1500);
  ws.onerror = () => ws.close();
}
function setText(el, v) {
  if (!el) return;
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
    setText(els.hud.audio, Math.round(p.alarm_band_db));
    setAudioBar(p.alarm_band_db);
  }
};

handlers["audio_alarm"] = (p) => {
  if (p.state === "alarm") {
    appendFeed("alarm", `audio alarm (${p.source || "?"}): ${p.reason || "tone detected"}`);
    // arm-state visual before the agent picks it up
    if (state.verdict === "idle") {
      setStatus("armed", "Alarm detected");
      advanceStepper("trigger");
    }
  }
};

handlers["agent_state"] = (p) => {
  if (p.mission_id && p.mission_id !== state.missionId) {
    state.missionId = p.mission_id;
    state.startedAt = Date.now();
    state.evidence  = 0;
    clearFeed();
    clearEvidence();
    els.mission.id.textContent      = p.mission_id;
    els.mission.trigger.textContent = p.trigger ? `trigger: ${p.trigger}` : "—";
    setBanner("running", "Mission active — inspecting the area",
      "The agent is autonomously sweeping the room. Verdict will appear here as soon as it submits a finding.",
      "MISSION " + p.mission_id);
    setDispatchBadge("Awaiting verdict", false);
    setStatus("running", "Mission active");
    setMissionStep("Spinning up");
    resetStepper();
    advanceStepper("inspect");
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

  // When the agent fires a terminal report_* tool, the decision moment
  // has arrived — bump the stepper to "decide".
  if (typeof p.tool === "string" && p.tool.startsWith("report_")) {
    advanceStepper("decide");
  }
};

handlers["agent_tool_result"] = (p) => {
  appendFeed("result", truncate(p.result || "", 200));
};

handlers["vision_result"] = (p) => {
  if (p.source !== "agent") return;
  const empty = els.evidence.gallery.querySelector(".evidence-empty");
  if (empty) empty.remove();
  els.evidence.gallery.appendChild(buildEvidenceTile(p));
  state.evidence += 1;
  els.evidence.count.textContent       = String(state.evidence);
  els.mission.evidenceCnt.textContent  = String(state.evidence);
  els.evidence.gallery.scrollLeft = els.evidence.gallery.scrollWidth;
};

handlers["incident"] = (p) => {
  stopTimer();
  const v = p.verdict || "unknown";
  state.verdict = v;
  const subtitleByVerdict = {
    real_fire:   "Fire confirmed. Fire department has been dispatched.",
    false_alarm: "No real fire detected — alarm logged with reasoning below.",
    unknown:     "Verdict inconclusive. Manual review recommended.",
  };
  setBanner(
    v,
    p.title || subtitleByVerdict[v] || "Incident",
    p.summary || subtitleByVerdict[v] || "",
    `${(p.evidence || []).length} evidence · ID ${p.incident_id}`,
  );

  // Dispatch badge — surfaces the notification status in its own slot.
  const dispatchByVerdict = {
    real_fire:   p.notified_dept ? "Fire department notified" : "Notification failed",
    false_alarm: "No dispatch — logged",
    unknown:     "Manual review",
  };
  setDispatchBadge(dispatchByVerdict[v] || (p.notified_dept ? "Dispatched" : "Logged"),
                   v === "real_fire" || v === "false_alarm" || v === "unknown");

  setStatus(v, p.title || v);
  setMissionStep(p.summary || p.title || "Mission complete");
  markStepperVerdict(v);

  // Verdict reasoning gets a colored block at the top of the feed —
  // the headline payoff moment of the dashboard. Remove any previous
  // block before appending the new one.
  els.mission.feed.querySelectorAll(".verdict-block").forEach((n) => n.remove());
  if (Array.isArray(p.reasons) && p.reasons.length) {
    const block = document.createElement("div");
    block.className = "verdict-block";
    block.dataset.verdict = v;
    const h = document.createElement("div");
    h.className = "vh";
    h.textContent = "Verdict reasoning";
    block.appendChild(h);
    const list = document.createElement("ul");
    for (const r of p.reasons) {
      const li = document.createElement("li");
      li.textContent = r;
      list.appendChild(li);
    }
    block.appendChild(list);
    els.mission.feed.prepend(block);
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

  // Severity stripe across the top of the tile
  const stripe = document.createElement("div");
  stripe.className = "sev-bar";
  tile.appendChild(stripe);

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

  // Relative timestamp from mission start (e.g. "+12.4 s")
  if (state.startedAt) {
    const t = document.createElement("span");
    t.className = "evidence-time";
    const dt = (Date.now() - state.startedAt) / 1000;
    t.textContent = `+${dt.toFixed(1)} s`;
    head.appendChild(t);
  }

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
setDispatchBadge("No dispatch", false);
clearEvidence();
resetStepper();
setBatteryRing(NaN);
setFlightRing(0);
setAltBar(0);
setAudioBar(AUDIO_FLOOR_DB);

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
