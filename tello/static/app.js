/* FireDrone Tello — dashboard client.
 *
 * Two WebSockets:
 *   /ws/telemetry  – server pushes status + telemetry @ ~5 Hz
 *   /ws/control    – client sends JSON commands; server replies per command.
 *                    If this WS drops while the drone is flying, the server
 *                    cuts the motors. So we auto-reconnect, but the operator
 *                    has to manually take off again.
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
};

const state = {
  connected: false,
  streaming: false,
  flying: false,
  controlWs: null,
};

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
  // Cache-bust the MJPEG URL so reconnects re-establish the stream cleanly.
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

  // Badges
  setBadge(els.conn,   snap.connected ? "good" : "gray",
                       snap.connected ? "connected" : "disconnected");
  setBadge(els.stream, snap.streaming ? "good" : "gray",
                       snap.streaming ? "streaming" : "stream off");
  setBadge(els.fly,    snap.flying ? "good" : "gray",
                       snap.flying ? "in air" : "grounded");

  // Telemetry numbers
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

  // Battery warning color
  if (typeof t.battery_pct === "number" && t.battery_pct > 0 && t.battery_pct < 15) {
    els.t.battery.classList.add("low-batt");
    setBadge(els.conn, "bad", `battery ${t.battery_pct}%`);
  } else {
    els.t.battery.classList.remove("low-batt");
  }

  // Status line
  els.status.textContent = snap.last_status || "—";

  // Error banner
  if (snap.last_error) {
    els.lastError.hidden = false;
    els.lastError.textContent = snap.last_error;
  } else {
    els.lastError.hidden = true;
  }

  // Side-effect: kick the video on/off
  if (state.streaming && !els.video.src) startVideo();
  if (!state.streaming && els.video.src) stopVideo();

  // Side-effect: log flight transitions
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

// ----------------------- control WebSocket --------------------------- //

function connectControlWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/control`;
  const ws = new WebSocket(url);
  state.controlWs = ws;

  ws.onopen = () => {
    log("info", "control channel open");
    // heartbeat so we know the WS round-trip works
    sendCommand({ action: "ping" });
  };
  ws.onclose = () => {
    log("warn", "control channel closed, reconnecting...");
    state.controlWs = null;
    setTimeout(connectControlWs, 1000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    try {
      const payload = JSON.parse(ev.data);
      handleControlResponse(payload);
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
  if (p.action && p.action !== "ping") {
    log("cmd", `${p.action} -> ${p.status || "ok"}`);
  }
}

function sendCommand(cmd) {
  const ws = state.controlWs;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    log("warn", `control channel not ready, dropped: ${cmd.action}`);
    return;
  }
  ws.send(JSON.stringify(cmd));
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
  } catch (err) {
    log("error", `connect failed: ${err}`);
  } finally {
    els.connect.disabled = false;
  }
});

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
  log("error", "EMERGENCY — cutting motors");
  sendCommand({ action: "emergency" });
});

// Wire every [data-action] button automatically.
document.querySelectorAll("[data-action]").forEach((btn) => {
  btn.addEventListener("click", () => {
    sendCommand(btnToCommand(btn));
  });
});

function btnToCommand(btn) {
  const action = btn.dataset.action;
  const direction = btn.dataset.direction;
  const cmd = { action };
  if (direction) cmd.direction = direction;
  return cmd;
}

// -------------------------- keyboard --------------------------------- //

const KEYMAP = {
  KeyT:        { action: "takeoff" },
  KeyL:        { action: "land" },
  Escape:      { action: "emergency" },
  KeyW:        { action: "move",   direction: "forward" },
  KeyS:        { action: "move",   direction: "back" },
  KeyA:        { action: "move",   direction: "left" },
  KeyD:        { action: "move",   direction: "right" },
  Space:       { action: "move",   direction: "up" },
  ControlLeft: { action: "move",   direction: "down" },
  ControlRight:{ action: "move",   direction: "down" },
  KeyQ:        { action: "rotate", direction: "ccw" },
  KeyE:        { action: "rotate", direction: "cw" },
  Digit1:      { action: "flip",   direction: "f" },
  Digit2:      { action: "flip",   direction: "b" },
  Digit3:      { action: "flip",   direction: "l" },
  Digit4:      { action: "flip",   direction: "r" },
};

window.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  // Ignore if user is typing in a form field.
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;

  const cmd = KEYMAP[e.code];
  if (!cmd) return;
  e.preventDefault();
  sendCommand(cmd);
});

// ----------------------------- boot ---------------------------------- //

log("info", "dashboard loaded");
connectTelemetryWs();
connectControlWs();
