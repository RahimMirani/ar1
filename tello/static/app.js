/* FireDrone Tello — dashboard client.
 *
 * Two WebSockets:
 *   /ws/telemetry  – server pushes status + telemetry @ ~5 Hz
 *   /ws/control    – client sends JSON commands; server replies per command.
 *                    If this WS drops while the drone is flying, the server
 *                    cuts the motors. So we auto-reconnect, but the operator
 *                    has to manually take off again.
 *
 * Live motion uses an RC velocity model: the user holds keys (arrows / Space /
 * Shift / Q / E) and we send a `set_velocity` command whenever the resulting
 * vector changes. The drone reacts within ~50 ms.
 *
 * Video is just <img src="/video.mjpg"> — the browser handles MJPEG natively.
 */

const $ = (id) => document.getElementById(id);

const els = {
  video:        $("video"),
  videoOverlay: $("video-overlay"),
  conn:         $("status-conn"),
  stream:       $("status-stream"),
  fly:          $("status-fly"),
  emergency:    $("btn-emergency"),
  connect:      $("btn-connect"),
  disconnect:   $("btn-disconnect"),
  takeoff:      $("btn-takeoff"),
  land:         $("btn-land"),
  lastError:    $("last-error"),
  status:       $("t-status"),
  log:          $("event-log"),
  clearLog:     $("btn-clear-log"),
  t: {
    battery:  $("t-battery"),
    tof:      $("t-tof"),
    height:   $("t-height"),
    flight:   $("t-flight"),
    yaw:      $("t-yaw"),
    pitch:    $("t-pitch"),
    roll:     $("t-roll"),
    sx:       $("t-sx"),
    sy:       $("t-sy"),
    sz:       $("t-sz"),
    temp:     $("t-temp"),
  },
  // operator-console additions (all optional — handlers null-check before use)
  mission: {
    pill:  $("mission-pill"),
    label: $("mission-label"),
  },
  ind: {
    conn:   $("ind-conn"),
    stream: $("ind-stream"),
    fly:    $("ind-fly"),
  },
  hud: {
    battery:  $("hud-battery"),
    altitude: $("hud-altitude"),
    flight:   $("hud-flight"),
    link:     $("hud-link"),
    feed:     $("hud-feed"),
    airborne: $("hud-airborne"),
    velocity: $("hud-velocity"),
    vx:       $("hud-vx"),
    vy:       $("hud-vy"),
    vz:       $("hud-vz"),
    vyaw:     $("hud-vyaw"),
    linkq:    $("hud-linkq"),
    snr:      $("hud-snr"),
    loss:     $("hud-loss"),
    rtt:      $("hud-rtt"),
    viderr:   $("hud-viderr"),
    fence:    $("hud-fence"),
  },
  vel: {
    box:  $("vel-readout"),
    vx:   $("vel-vx"),
    vy:   $("vel-vy"),
    vz:   $("vel-vz"),
    yaw:  $("vel-yaw"),
  },
  batteryBar: $("battery-bar"),

  vision: {
    btn:      $("btn-vision-analyze"),
    result:   $("vision-result"),
    severity: $("vision-severity"),
    chips:    $("vision-chips"),
    conf:     $("vision-conf"),
    desc:     $("vision-desc"),
    reasons:  $("vision-reasons"),
    thumb:    $("vision-thumb"),
    meta:     $("vision-meta"),
    error:    $("vision-error"),
  },

  audio: {
    start:    $("btn-audio-start"),
    stop:     $("btn-audio-stop"),
    simulate: $("btn-audio-simulate"),
    badge:    $("audio-state-badge"),
    band:     $("audio-meter-band"),
    bandDb:   $("audio-meter-band-db"),
    broad:    $("audio-meter-broad"),
    broadDb:  $("audio-meter-broad-db"),
    device:   $("audio-device"),
    error:    $("audio-error"),
  },
};

const state = {
  connected: false,
  streaming: false,
  flying: false,
  controlWs: null,
  emergency: false,
  fenceTier: "ok",
};

// Human-readable label per fence tier, surfaced both on the HUD inset and
// in the event log when the tier changes.
const FENCE_LABELS = {
  ok:      "link ok",
  caution: "caution",
  hover:   "auto-hover",
  land:    "auto-land",
};

// Map data-hold-key value -> array of matching button elements, so we can
// add/remove the .held class when the operator presses/releases.
const holdKeyButtons = new Map();
document.querySelectorAll("[data-hold-key]").forEach((btn) => {
  const code = btn.dataset.holdKey;
  if (!holdKeyButtons.has(code)) holdKeyButtons.set(code, []);
  holdKeyButtons.get(code).push(btn);
});

function setButtonHeld(code, on) {
  const btns = holdKeyButtons.get(code);
  if (!btns) return;
  for (const b of btns) b.classList.toggle("held", !!on);
}

// Mirror the connected/streaming/flying/emergency flags onto the mission
// pill and indicator dots. Single source of truth for the redesigned chrome.
function updateMissionState() {
  let stateKey = "idle";
  let label    = "Idle";
  if (state.emergency)      { stateKey = "emergency"; label = "Emergency"; }
  else if (state.flying)    { stateKey = "flight";    label = "In Flight"; }
  else if (state.connected) { stateKey = "armed";     label = "Armed"; }

  if (els.mission.pill)  els.mission.pill.dataset.state = stateKey;
  if (els.mission.label) els.mission.label.textContent  = label;

  setIndicator(els.ind.conn,   state.connected);
  setIndicator(els.ind.stream, state.streaming);
  setIndicator(els.ind.fly,    state.flying);
  setIndicator(els.hud.link,     state.connected);
  setIndicator(els.hud.feed,     state.streaming);
  setIndicator(els.hud.airborne, state.flying);
}

function setIndicator(el, on) {
  if (!el) return;
  el.dataset.on = on ? "true" : "false";
}

// ------------------------------ logging ------------------------------- //

const MAX_LOG_LINES = 200;

function ts() {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function log(level, msg) {
  const li = document.createElement("li");
  li.innerHTML =
    `<span class="ts">${ts()}</span>` +
    `<span class="lvl lvl-${level}">${level}</span>` +
    `<span class="msg"></span>`;
  li.querySelector(".msg").textContent = msg;
  els.log.prepend(li);
  while (els.log.children.length > MAX_LOG_LINES) {
    els.log.removeChild(els.log.lastChild);
  }
}

els.clearLog.addEventListener("click", () => {
  els.log.replaceChildren();
});

// --------------------------- video plumbing --------------------------- //

function startVideo() {
  els.video.src = `/video.mjpg?t=${Date.now()}`;
  els.video.onload = () => els.videoOverlay.classList.add("hidden");
  els.video.onerror = () => els.videoOverlay.classList.remove("hidden");
}

function stopVideo() {
  els.video.removeAttribute("src");
  els.videoOverlay.classList.remove("hidden");
}

// ---------------------- telemetry WebSocket -------------------------- //

function connectTelemetryWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/telemetry`;
  const ws = new WebSocket(url);
  ws.onopen = () => log("info", "telemetry stream connected");
  ws.onclose = () => {
    log("warn", "telemetry stream disconnected, retrying...");
    setTimeout(connectTelemetryWs, 1000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    try {
      handleTelemetry(JSON.parse(ev.data));
    } catch (err) {
      log("error", `bad telemetry payload: ${err}`);
    }
  };
}

function handleTelemetry(snap) {
  const wasFlying = state.flying;
  state.connected = !!snap.connected;
  state.streaming = !!snap.streaming;
  state.flying    = !!snap.flying;

  setBadge(els.conn,   snap.connected ? "good" : "gray",
                       snap.connected ? "connected" : "disconnected");
  setBadge(els.stream, snap.streaming ? "good" : "gray",
                       snap.streaming ? "streaming" : "stream off");
  setBadge(els.fly,    snap.flying ? "good" : "gray",
                       snap.flying ? "in air" : "grounded");

  const t = snap.telemetry || {};
  setNum(els.t.battery, t.battery_pct);
  setNum(els.t.tof,     t.tof_cm);
  setNum(els.t.height,  t.height_cm);
  setNum(els.t.flight,  t.flight_time_s);
  setNum(els.t.yaw,     t.yaw_deg);
  setNum(els.t.pitch,   t.pitch_deg);
  setNum(els.t.roll,    t.roll_deg);
  setNum(els.t.sx,      t.speed_x);
  setNum(els.t.sy,      t.speed_y);
  setNum(els.t.sz,      t.speed_z);
  setNum(els.t.temp,    t.temperature_c);

  if (typeof t.battery_pct === "number" && t.battery_pct > 0 && t.battery_pct < 15) {
    els.t.battery.classList.add("low-batt");
    setBadge(els.conn, "bad", `battery ${t.battery_pct}%`);
  } else {
    els.t.battery.classList.remove("low-batt");
  }

  updateBatteryBar(t.battery_pct);
  mirrorHud(t);
  updateMissionState();

  els.status.textContent = snap.last_status || "—";

  if (snap.last_error) {
    els.lastError.hidden = false;
    els.lastError.textContent = snap.last_error;
  } else {
    els.lastError.hidden = true;
  }

  if (state.streaming && !els.video.src) startVideo();
  if (!state.streaming && els.video.src) stopVideo();

  if (!wasFlying && state.flying) log("info", "drone is now in air");
  if (wasFlying && !state.flying) log("info", "drone has landed / motors stopped");
}

function setBadge(el, cls, text) {
  el.className = `badge badge-${cls}`;
  el.textContent = text;
}

function setNum(el, v) {
  if (v === null || v === undefined) {
    el.textContent = "—";
    return;
  }
  if (typeof v === "number" && !Number.isInteger(v)) {
    el.textContent = v.toFixed(1);
  } else {
    el.textContent = v;
  }
}

function updateBatteryBar(pct) {
  if (!els.batteryBar) return;
  const fill = els.batteryBar.querySelector(".fill");
  if (!fill) return;
  if (typeof pct !== "number" || pct < 0) {
    fill.style.width = "0%";
    els.batteryBar.classList.remove("low");
    return;
  }
  const clamped = Math.max(0, Math.min(100, pct));
  fill.style.width = `${clamped}%`;
  els.batteryBar.classList.toggle("low", clamped > 0 && clamped < 15);
}

// Mirror a subset of telemetry into the glass HUD overlay on the video.
// Uses the same source data as the side-pane telemetry grid so the two
// stay in lockstep at the 5 Hz the server pushes.
function mirrorHud(t) {
  setNum(els.hud.battery,  t.battery_pct);
  setNum(els.hud.altitude, t.tof_cm);
  setNum(els.hud.flight,   t.flight_time_s);
  paintLink(t);
}

// Paint the four link-health numbers into the bottom-right HUD inset.
// All four can be null (server hasn't sampled yet, or the SDK 'wifi?'
// query timed out) — render '—' so the inset doesn't flash blank. The
// fence tier on the snapshot decides the border color and the status
// line at the bottom of the inset.
function paintLink(t) {
  setNum(els.hud.snr,    t.wifi_snr_db);
  setNum(els.hud.loss,   t.packet_loss_pct);
  setNum(els.hud.rtt,    t.link_rtt_ms);
  setNum(els.hud.viderr, t.video_errors_per_sec);

  const tier = typeof t.link_fence === "string" ? t.link_fence : "ok";
  if (els.hud.linkq) els.hud.linkq.dataset.fence = tier;
  if (els.hud.fence) els.hud.fence.textContent = FENCE_LABELS[tier] || tier;

  if (tier !== state.fenceTier) {
    const prev = state.fenceTier;
    state.fenceTier = tier;
    const lvl = (tier === "land" || tier === "hover") ? "error"
              : (tier === "caution") ? "warn"
              : "info";
    log(lvl, `link fence: ${prev} -> ${tier}`);
  }
}

// ----------------------- control WebSocket --------------------------- //

function connectControlWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/control`;
  const ws = new WebSocket(url);
  state.controlWs = ws;

  ws.onopen = () => {
    log("info", "control channel open");
    sendCommand({ action: "ping" });
  };
  ws.onclose = () => {
    log("warn", "control channel closed, reconnecting...");
    state.controlWs = null;
    clearHeldKeys();
    setTimeout(connectControlWs, 1000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    try {
      handleControlResponse(JSON.parse(ev.data));
    } catch (err) {
      log("error", `bad control payload: ${err}`);
    }
  };
}

function handleControlResponse(p) {
  if (p.ok === false) {
    log("error", p.error || "command failed");
    return;
  }
  if (p.silent) return;
  if (p.action && p.action !== "ping") {
    log("cmd", `${p.action} -> ${p.status || "ok"}`);
  }
}

function sendCommand(cmd) {
  const ws = state.controlWs;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    if (cmd.action !== "set_velocity") {
      log("warn", `control channel not ready, dropped: ${cmd.action}`);
    }
    return;
  }
  ws.send(JSON.stringify(cmd));
}

// -------------------- velocity (RC) state machine -------------------- //
//
// We track which "hold keys" are currently active (either real keyboard keys
// or synthetic ones from on-screen button mousedown/touchstart). Whenever the
// set changes, we recompute the resulting velocity vector and send it.

const VELOCITY = {
  LR:  60,   // ± left / right
  FB:  60,   // ± back / forward
  UD:  50,   // ± down / up
  YAW: 70,   // ± yaw left / right
};

const HOLD_KEYS = new Set([
  "KeyW", "KeyS", "KeyA", "KeyD",
  "Space", "ShiftLeft", "ShiftRight",
  "KeyQ", "KeyE",
]);

const heldKeys = new Set();
let   lastSentVel = { lr: 0, fb: 0, ud: 0, yaw: 0 };

function recomputeVelocity() {
  let lr = 0, fb = 0, ud = 0, yaw = 0;

  if (heldKeys.has("KeyW")) fb += VELOCITY.FB;
  if (heldKeys.has("KeyS")) fb -= VELOCITY.FB;
  if (heldKeys.has("KeyD")) lr += VELOCITY.LR;
  if (heldKeys.has("KeyA")) lr -= VELOCITY.LR;
  if (heldKeys.has("Space"))      ud += VELOCITY.UD;
  if (heldKeys.has("ShiftLeft") || heldKeys.has("ShiftRight")) ud -= VELOCITY.UD;
  if (heldKeys.has("KeyE"))       yaw += VELOCITY.YAW;
  if (heldKeys.has("KeyQ"))       yaw -= VELOCITY.YAW;

  if (lr === lastSentVel.lr && fb === lastSentVel.fb &&
      ud === lastSentVel.ud && yaw === lastSentVel.yaw) {
    return;
  }
  lastSentVel = { lr, fb, ud, yaw };
  sendCommand({ action: "set_velocity", lr, fb, ud, yaw });
  paintVelocity(lr, fb, ud, yaw);
}

// Paint the velocity vector into the side-pane readout and the video HUD
// inset. Maps lr -> vx (right+), fb -> vy (forward+), ud -> vz (up+),
// yaw -> ω. Same data the drone gets; we just surface it visually so the
// operator can see what's been sent.
function paintVelocity(lr, fb, ud, yaw) {
  if (els.vel.vx)  els.vel.vx.textContent  = lr;
  if (els.vel.vy)  els.vel.vy.textContent  = fb;
  if (els.vel.vz)  els.vel.vz.textContent  = ud;
  if (els.vel.yaw) els.vel.yaw.textContent = yaw;
  if (els.hud.vx)   els.hud.vx.textContent   = lr;
  if (els.hud.vy)   els.hud.vy.textContent   = fb;
  if (els.hud.vz)   els.hud.vz.textContent   = ud;
  if (els.hud.vyaw) els.hud.vyaw.textContent = yaw;

  const moving = (lr || fb || ud || yaw) !== 0;
  if (els.vel.box)      els.vel.box.classList.toggle("active", moving);
  if (els.hud.velocity) els.hud.velocity.classList.toggle("active", moving);
}

function holdKeyDown(code) {
  if (!HOLD_KEYS.has(code) || heldKeys.has(code)) return;
  heldKeys.add(code);
  setButtonHeld(code, true);
  recomputeVelocity();
}

function holdKeyUp(code) {
  if (!heldKeys.has(code)) return;
  heldKeys.delete(code);
  setButtonHeld(code, false);
  recomputeVelocity();
}

function clearHeldKeys() {
  if (heldKeys.size === 0) return;
  for (const code of heldKeys) setButtonHeld(code, false);
  heldKeys.clear();
  recomputeVelocity();
}

// ------------------------- button wiring ----------------------------- //

els.connect.addEventListener("click", async () => {
  log("info", "connecting...");
  els.connect.disabled = true;
  try {
    const res = await fetch("/api/connect", { method: "POST" });
    const data = await res.json();
    if (data.error) log("error", data.error);
    else log("info", "connect ok");
    reportLinkDiagnostics(data.link_diagnostics);
  } catch (err) {
    log("error", `connect failed: ${err}`);
  } finally {
    els.connect.disabled = false;
  }
});

// Surface any failed link-safety check in the event log. On a healthy
// connect everything is true and we stay silent. A False entry means a
// monkey-patch did not apply or a background thread did not start — a
// regression introduced by an edit to drone.py.
function reportLinkDiagnostics(diag) {
  if (!diag || typeof diag !== "object") return;
  const failed = Object.entries(diag)
    .filter(([, ok]) => ok === false)
    .map(([name]) => name);
  if (failed.length === 0) {
    log("info", "link safety checks: all pass");
  } else {
    log("error", `link safety degraded: ${failed.join(", ")}`);
  }
}

els.disconnect.addEventListener("click", async () => {
  log("info", "disconnecting...");
  try {
    await fetch("/api/disconnect", { method: "POST" });
    stopVideo();
  } catch (err) {
    log("error", `disconnect failed: ${err}`);
  }
});

els.emergency.addEventListener("click", () => {
  triggerEmergencyUi();
  sendCommand({ action: "emergency" });
});

// Visual-only side effect of the emergency action: log it, drop all held
// keys, flip the mission pill red, then auto-clear the flag after a few
// seconds so the pill returns to idle/armed once the operator regains
// control. The actual motor cut is handled server-side by the command.
function triggerEmergencyUi() {
  log("error", "EMERGENCY — cutting motors");
  clearHeldKeys();
  state.emergency = true;
  updateMissionState();
  clearTimeout(triggerEmergencyUi._t);
  triggerEmergencyUi._t = setTimeout(() => {
    state.emergency = false;
    updateMissionState();
  }, 4000);
}

// Discrete one-shot actions (takeoff, land, flips).
document.querySelectorAll("[data-action]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const cmd = { action: btn.dataset.action };
    if (btn.dataset.direction) cmd.direction = btn.dataset.direction;
    sendCommand(cmd);
  });
});

// Hold-to-fly on-screen buttons. mousedown / touchstart synthesize a held
// key; mouseup / mouseleave / touchend release it. We use pointer events so
// the same handler works for mouse, touch, and pen.
document.querySelectorAll("[data-hold-key]").forEach((btn) => {
  const code = btn.dataset.holdKey;
  const press   = (e) => { e.preventDefault(); btn.setPointerCapture?.(e.pointerId); holdKeyDown(code); };
  const release = (e) => { e.preventDefault(); btn.releasePointerCapture?.(e.pointerId); holdKeyUp(code); };
  btn.addEventListener("pointerdown",   press);
  btn.addEventListener("pointerup",     release);
  btn.addEventListener("pointercancel", release);
  btn.addEventListener("pointerleave",  (e) => { if (heldKeys.has(code)) holdKeyUp(code); });
});

// -------------------------- keyboard --------------------------------- //

// One-shot actions only fire on keydown (not repeat).
const ONE_SHOT_KEYS = {
  KeyT:   { action: "takeoff" },
  KeyL:   { action: "land" },
  Escape: { action: "emergency" },
  Digit1: { action: "flip", direction: "f" },
  Digit2: { action: "flip", direction: "b" },
  Digit3: { action: "flip", direction: "l" },
  Digit4: { action: "flip", direction: "r" },
};

window.addEventListener("keydown", (e) => {
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;

  if (HOLD_KEYS.has(e.code)) {
    e.preventDefault();
    if (e.repeat) return;
    holdKeyDown(e.code);
    return;
  }

  const oneShot = ONE_SHOT_KEYS[e.code];
  if (oneShot && !e.repeat) {
    e.preventDefault();
    if (oneShot.action === "emergency") triggerEmergencyUi();
    sendCommand(oneShot);
  }
});

window.addEventListener("keyup", (e) => {
  if (HOLD_KEYS.has(e.code)) {
    e.preventDefault();
    holdKeyUp(e.code);
  }
});

// Safety: if the browser tab loses focus or visibility, release everything
// so the drone stops instead of flying away unattended.
window.addEventListener("blur", clearHeldKeys);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearHeldKeys();
});

// ====================================================================
// AI event channel — vision, audio, perception, agent, notifier all
// fan out over the same /ws/events stream. We register a handler per
// event.type and dispatch.
// ====================================================================

const eventHandlers = {};

function connectEventsWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/events`;
  const ws = new WebSocket(url);
  ws.onopen  = () => log("info", "events stream connected");
  ws.onclose = () => {
    log("warn", "events stream disconnected, retrying...");
    setTimeout(connectEventsWs, 1000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    let payload;
    try { payload = JSON.parse(ev.data); }
    catch (err) { log("error", `bad event payload: ${err}`); return; }
    const fn = eventHandlers[payload.type];
    if (fn) fn(payload);
  };
}

// ----------------------------- vision ----------------------------- //

function renderVisionResult(p) {
  if (!els.vision.result) return;
  els.vision.result.hidden = false;
  els.vision.error.hidden  = true;

  const sev = p.severity || "none";
  els.vision.severity.dataset.severity = sev;
  els.vision.severity.textContent      = sev;

  els.vision.chips.replaceChildren();
  const addChip = (label, on, red) => {
    const s = document.createElement("span");
    s.className = "vision-chip" + (on ? (red ? " on-red" : " on") : "");
    s.textContent = label;
    els.vision.chips.appendChild(s);
  };
  addChip("fire",  !!p.fire_visible, true);
  addChip("smoke", !!p.smoke_visible, false);

  const cpct = Math.round((p.confidence || 0) * 100);
  els.vision.conf.textContent = `${cpct}% conf`;

  els.vision.desc.textContent = p.description || "";

  els.vision.reasons.replaceChildren();
  for (const r of (p.reasons || [])) {
    const li = document.createElement("li");
    li.textContent = r;
    els.vision.reasons.appendChild(li);
  }

  if (p.thumbnail_b64) {
    els.vision.thumb.src = `data:image/jpeg;base64,${p.thumbnail_b64}`;
    els.vision.thumb.hidden = false;
  } else {
    els.vision.thumb.hidden = true;
  }

  const src = p.source || "manual";
  els.vision.meta.textContent = `${p.model || "?"} · ${src} · ${p.latency_ms ?? "?"} ms`;

  const lvl = (p.fire_visible || sev === "high") ? "error"
            : (p.smoke_visible || sev === "medium") ? "warn"
            : "info";
  log(lvl, `vision (${src}): ${sev}${p.fire_visible ? " · fire" : ""}${p.smoke_visible ? " · smoke" : ""}`);
}

eventHandlers["vision_result"] = renderVisionResult;

// ----------------------------- audio ----------------------------- //

// Map a dBFS reading to a 0-100% bar width. -60 dB -> 0%, 0 dB -> 100%.
function dbToPct(db) {
  if (db === null || db === undefined || !isFinite(db)) return 0;
  return Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
}

function setAudioState(stateKey) {
  if (!els.audio.badge) return;
  els.audio.badge.dataset.state = stateKey;
  const labels = { idle: "off", armed: "listening", alarm: "ALARM", error: "error" };
  els.audio.badge.textContent = labels[stateKey] || stateKey;
  // Bar color follows state.
  for (const bar of [els.audio.band, els.audio.broad]) {
    if (!bar) continue;
    bar.classList.toggle("hot",   stateKey === "armed");
    bar.classList.toggle("alarm", stateKey === "alarm");
  }
}

function renderAudioLevel(p) {
  if (els.audio.band) {
    els.audio.band.querySelector(".fill").style.width = `${dbToPct(p.alarm_band_db)}%`;
    els.audio.bandDb.textContent = (typeof p.alarm_band_db === "number")
      ? `${p.alarm_band_db.toFixed(0)} dB` : "—";
  }
  if (els.audio.broad) {
    els.audio.broad.querySelector(".fill").style.width = `${dbToPct(p.broadband_db)}%`;
    els.audio.broadDb.textContent = (typeof p.broadband_db === "number")
      ? `${p.broadband_db.toFixed(0)} dB` : "—";
  }
}

function renderAudioAlarm(p) {
  setAudioState(p.state);
  const src = p.source ? ` [${p.source}]` : "";
  if (p.state === "alarm") {
    log("error", `audio alarm${src}: ${p.reason || "tone detected"}`);
  } else if (p.state === "armed") {
    log("info",  `audio armed${src}${p.reason ? " — " + p.reason : ""}`);
  } else if (p.state === "idle") {
    log("info",  "audio monitor stopped");
  } else if (p.state === "error") {
    log("error", `audio error: ${p.error || "unknown"}`);
    if (els.audio.error) {
      els.audio.error.hidden = false;
      els.audio.error.textContent = p.error || "audio error";
    }
  }
}

eventHandlers["audio_level"] = renderAudioLevel;
eventHandlers["audio_alarm"] = renderAudioAlarm;

async function postAudio(path) {
  try {
    const res = await fetch(path, { method: "POST" });
    const data = await res.json();
    if (data.device) els.audio.device.textContent = data.device;
    if (data.state)  setAudioState(data.state);
    if (data.error) {
      els.audio.error.hidden = false;
      els.audio.error.textContent = data.error;
      log("error", `audio: ${data.error}`);
    } else {
      els.audio.error.hidden = true;
    }
    return data;
  } catch (err) {
    log("error", `audio request failed: ${err}`);
    return null;
  }
}

if (els.audio.start)    els.audio.start.addEventListener("click",    () => postAudio("/api/audio/start"));
if (els.audio.stop)     els.audio.stop.addEventListener("click",     () => postAudio("/api/audio/stop"));
if (els.audio.simulate) els.audio.simulate.addEventListener("click", () => postAudio("/api/audio/simulate"));

if (els.vision.btn) {
  els.vision.btn.addEventListener("click", async () => {
    els.vision.btn.disabled = true;
    els.vision.btn.textContent = "Analyzing...";
    els.vision.error.hidden = true;
    try {
      const res = await fetch("/api/vision/analyze", { method: "POST" });
      const data = await res.json();
      if (data.error) {
        els.vision.error.hidden = false;
        els.vision.error.textContent = data.error;
        log("error", `vision: ${data.error}`);
      } else {
        renderVisionResult({ source: "manual", ...data });
      }
    } catch (err) {
      els.vision.error.hidden = false;
      els.vision.error.textContent = String(err);
      log("error", `vision: ${err}`);
    } finally {
      els.vision.btn.disabled = false;
      els.vision.btn.textContent = "Analyze current view";
    }
  });
}

// ----------------------------- boot ---------------------------------- //

log("info", "dashboard loaded");
connectTelemetryWs();
connectControlWs();
connectEventsWs();
