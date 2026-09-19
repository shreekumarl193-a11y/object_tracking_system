const BACKEND_WS = "ws://localhost:8000/ws/stream";
const BACKEND_API = "http://localhost:8000/api";

const videoImg = document.getElementById("video-stream");
const canvas = document.getElementById("overlay-canvas");
const ctx = canvas.getContext("2d");

// Movement Graph Canvas
const graphCanvas = document.getElementById("movement-graph-canvas");
const graphCtx = graphCanvas.getContext("2d");
let activeGraphMetric = "displacement"; // "displacement" or "speed"
let graphDataHistory = [];

// Header Metrics & Badges
const fpsVal = document.getElementById("fps-val");
const trackerVal = document.getElementById("tracker-val");
const playbackPill = document.getElementById("playback-pill");
const classPill = document.getElementById("class-pill");

// Vector Display ("From Where to Where")
const targetNameDisplay = document.getElementById("target-name-display");
const startCoordVal = document.getElementById("start-coord-val");
const currCoordVal = document.getElementById("curr-coord-val");
const dispVal = document.getElementById("disp-val");
const distVal = document.getElementById("dist-val");
const speedVal = document.getElementById("speed-val");
const dirVal = document.getElementById("dir-val");

// Timeline & Finished Banner
const timelineFill = document.getElementById("timeline-fill");
const timelineText = document.getElementById("timeline-text");
const finishedBanner = document.getElementById("video-finished-banner");
const bannerBtnReset = document.getElementById("banner-btn-reset");
const bannerBtnExport = document.getElementById("banner-btn-export");

// Data Table & Logs
const telemetryBody = document.getElementById("telemetry-table-body");
const recordsCount = document.getElementById("records-count");
const logStream = document.getElementById("log-stream");

let ws = null;
let currentDetections = [];
let isDrawingROI = false;
let startX = 0, startY = 0, currentX = 0, currentY = 0;

function initWebSocket() {
  ws = new WebSocket(BACKEND_WS);

  ws.onopen = () => appendLog("Connected to tracking WebSocket on port 8000.");

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    videoImg.src = data.image;

    const t = data.telemetry;
    const pb = data.playback_info;

    fpsVal.innerText = t.fps.toFixed(1);
    trackerVal.innerText = t.active_tracker;
    
    // Playback State
    if (pb.video_ended) {
      playbackPill.innerText = "FINISHED";
      playbackPill.className = "status-pill status-finished";
      finishedBanner.style.display = "flex";
    } else if (pb.is_paused) {
      playbackPill.innerText = "PAUSED";
      playbackPill.className = "status-pill status-paused";
      finishedBanner.style.display = "none";
    } else {
      playbackPill.innerText = "PLAYING";
      playbackPill.className = "status-pill status-stable";
      finishedBanner.style.display = "none";
    }

    // Timeline Update
    if (pb.source_type !== "webcam" && pb.total_frames > 0) {
      const pct = Math.min(100, Math.round((pb.current_frame / pb.total_frames) * 100));
      timelineFill.style.width = `${pct}%`;
      timelineText.innerText = `Frame: ${pb.current_frame} / ${pb.total_frames} (${pct}%)`;
    } else {
      timelineFill.style.width = "100%";
      timelineText.innerText = `Live Webcam Feed (Frame #${pb.current_frame})`;
    }

    // Object Identification & Vector Display
    classPill.innerText = t.object_class ? t.object_class.toUpperCase() : "NONE";
    targetNameDisplay.innerText = t.object_name || "Unselected";

    if (t.start_point && t.start_point.length >= 2) {
      startCoordVal.innerText = `(${t.start_point[0]}, ${t.start_point[1]})`;
    } else {
      startCoordVal.innerText = "( - , - )";
    }

    if (t.current_point && t.current_point.length >= 2) {
      currCoordVal.innerText = `(${t.current_point[0]}, ${t.current_point[1]})`;
    } else {
      currCoordVal.innerText = "( - , - )";
    }

    dispVal.innerText = `${t.displacement_px} px`;
    distVal.innerText = `${t.cumulative_dist_px} px`;
    speedVal.innerText = `${t.speed_px_s} px/s`;
    dirVal.innerText = `${t.direction} (${t.angle_deg}°)`;

    // Rebuild the graph from backend telemetry, including the final video frame.
    const telemetryRows = Array.isArray(t.recent_rows) ? t.recent_rows : [];
    graphDataHistory = telemetryRows.map(row => ({
      frame: row.frame_id,
      displacement: Number(row.displacement_px) || 0,
      speed: Number(row.speed_px_s) || 0
    }));
    renderMovementGraph();

    // Telemetry Table
    if (telemetryRows.length > 0) {
      renderDataTable(telemetryRows);
      recordsCount.innerText = `${telemetryRows.length} records${pb.video_ended ? " (final)" : ""}`;
    } else {
      telemetryBody.innerHTML = '<tr><td colspan="9" class="empty-state">No tracking data collected yet. Select one target.</td></tr>';
      recordsCount.innerText = "0 records";
    }

    // Event Logs
    if (t.logs && t.logs.length > 0) {
      logStream.innerHTML = "";
      t.logs.forEach(msg => appendLog(msg));
    }

    currentDetections = data.detections || [];
    drawOverlays(data.is_tracking, data.target_bbox, t);
  };

  ws.onerror = () => appendLog("WebSocket connection error. Check if backend is active.");
  ws.onclose = () => setTimeout(initWebSocket, 2000);
}

// Live Canvas Movement Graph Plotter
function renderMovementGraph() {
  const w = graphCanvas.width;
  const h = graphCanvas.height;
  graphCtx.clearRect(0, 0, w, h);

  if (graphDataHistory.length < 2) {
    graphCtx.fillStyle = "#64748b";
    graphCtx.font = "11px Inter, sans-serif";
    graphCtx.textAlign = "center";
    graphCtx.fillText("Select an object or click 'Quick Target' to stream graph curves", w / 2, h / 2);
    return;
  }

  const padLeft = 45;
  const padBottom = 25;
  const plotW = w - padLeft - 15;
  const plotH = h - padBottom - 10;

  const values = graphDataHistory.map(d => activeGraphMetric === "displacement" ? d.displacement : d.speed);
  const maxVal = Math.max(Math.max(...values), 20.0);

  // Gridlines & Labels
  graphCtx.strokeStyle = "#334155";
  graphCtx.lineWidth = 0.8;
  graphCtx.setLineDash([]);

  const steps = 3;
  for (let s = 0; s <= steps; s++) {
    const yVal = (maxVal / steps) * s;
    const yPos = h - padBottom - (s / steps) * plotH;

    graphCtx.beginPath();
    graphCtx.moveTo(padLeft, yPos);
    graphCtx.lineTo(w - 10, yPos);
    graphCtx.stroke();

    graphCtx.fillStyle = "#94a3b8";
    graphCtx.font = "9px 'JetBrains Mono', monospace";
    graphCtx.textAlign = "right";
    graphCtx.fillText(`${Math.round(yVal)}`, padLeft - 6, yPos + 3);
  }
  graphCtx.setLineDash([]);

  // Plot Curve
  const color = activeGraphMetric === "displacement" ? "#06b6d4" : "#f59e0b";
  graphCtx.strokeStyle = color;
  graphCtx.lineWidth = 2.2;
  graphCtx.beginPath();

  const points = [];
  for (let i = 0; i < values.length; i++) {
    const xPos = padLeft + (i / (values.length - 1)) * plotW;
    const yPos = h - padBottom - (values[i] / maxVal) * plotH;
    points.push({ x: xPos, y: yPos });
    if (i === 0) graphCtx.moveTo(xPos, yPos);
    else graphCtx.lineTo(xPos, yPos);
  }
  graphCtx.stroke();

  // Shaded Area
  graphCtx.lineTo(points[points.length - 1].x, h - padBottom);
  graphCtx.lineTo(points[0].x, h - padBottom);
  graphCtx.closePath();
  const grad = graphCtx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, activeGraphMetric === "displacement" ? "rgba(6, 182, 212, 0.3)" : "rgba(245, 158, 11, 0.3)");
  grad.addColorStop(1, "rgba(15, 23, 42, 0.0)");
  graphCtx.fillStyle = grad;
  graphCtx.fill();

  // Head Marker
  const lastPt = points[points.length - 1];
  graphCtx.fillStyle = color;
  graphCtx.beginPath();
  graphCtx.arc(lastPt.x, lastPt.y, 4, 0, 2 * Math.PI);
  graphCtx.fill();

  graphCtx.fillStyle = color;
  graphCtx.font = "bold 10px 'JetBrains Mono', monospace";
  graphCtx.textAlign = "left";
  const unit = activeGraphMetric === "displacement" ? "px" : "px/s";
  graphCtx.fillText(`${values[values.length - 1].toFixed(1)} ${unit}`, lastPt.x - 45, Math.max(16, lastPt.y - 8));
}

// Graph Tab Handlers
document.getElementById("tab-displacement").addEventListener("click", (e) => {
  activeGraphMetric = "displacement";
  e.target.classList.add("active");
  document.getElementById("tab-speed").classList.remove("active");
  renderMovementGraph();
});

document.getElementById("tab-speed").addEventListener("click", (e) => {
  activeGraphMetric = "speed";
  e.target.classList.add("active");
  document.getElementById("tab-displacement").classList.remove("active");
  renderMovementGraph();
});

// Canvas Overlays & Trajectory Rendering
function drawOverlays(isTracking, targetBbox, telemetry) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Glowing Trajectory Path
  if (isTracking && telemetry.trajectory && telemetry.trajectory.length > 1) {
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (let i = 1; i < telemetry.trajectory.length; i++) {
      const prev = telemetry.trajectory[i - 1];
      const curr = telemetry.trajectory[i];
      const alpha = (i / telemetry.trajectory.length).toFixed(2);
      ctx.strokeStyle = `rgba(6, 182, 212, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(prev[0], prev[1]);
      ctx.lineTo(curr[0], curr[1]);
      ctx.stroke();
    }

    // Origin Dot
    const origin = telemetry.trajectory[0];
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath();
    ctx.arc(origin[0], origin[1], 5, 0, 2 * Math.PI);
    ctx.fill();

    // Head Dot
    const last = telemetry.trajectory[telemetry.trajectory.length - 1];
    ctx.fillStyle = "#06b6d4";
    ctx.beginPath();
    ctx.arc(last[0], last[1], 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  // 2. Active Bounding Box
  if (isTracking && targetBbox) {
    const [x, y, w, h] = targetBbox;
    const isCSRT = telemetry.active_tracker === "CSRT";
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = isCSRT ? "#06b6d4" : "#10b981";
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = isCSRT ? "#06b6d4" : "#10b981";
    ctx.fillRect(x, Math.max(0, y - 22), 170, 22);
    ctx.fillStyle = "#000000";
    ctx.font = "bold 11px Inter, sans-serif";
    ctx.fillText(`${telemetry.object_name} [${telemetry.active_tracker}]`, x + 6, Math.max(15, y - 6));
  } else if (!isTracking) {
    // 3. Detected Candidate Boxes
    currentDetections.forEach((det) => {
      const [x, y, w, h] = det.bbox;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(168, 85, 247, 0.8)";
      ctx.setLineDash([]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      ctx.fillStyle = "rgba(168, 85, 247, 0.85)";
      ctx.fillRect(x, Math.max(0, y - 18), 120, 18);
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px Inter, sans-serif";
      ctx.fillText(`${det.class_name} (${(det.confidence * 100).toFixed(0)}%)`, x + 4, Math.max(13, y - 5));
    });
  }

  // 4. Drag ROI Box
  if (isDrawingROI) {
    const rx = Math.min(startX, currentX);
    const ry = Math.min(startY, currentY);
    const rw = Math.abs(currentX - startX);
    const rh = Math.abs(currentY - startY);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#f59e0b";
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = "rgba(245, 158, 11, 0.2)";
    ctx.fillRect(rx, ry, rw, rh);
  }
}

// Render Data Table Rows
function renderDataTable(rows) {
  telemetryBody.innerHTML = "";
  rows.slice().reverse().forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.frame_id}</td>
      <td style="color:#a855f7; font-weight:bold;">${r.object_name}</td>
      <td style="color:#f43f5e;">${r.start_point}</td>
      <td style="color:#06b6d4;">${r.current_point}</td>
      <td>${r.displacement_px} px</td>
      <td style="color:#f59e0b;">${r.speed_px_s} px/s</td>
      <td>${r.direction}</td>
      <td style="color:#10b981;">${r.active_tracker}</td>
      <td>${r.status}</td>
    `;
    telemetryBody.appendChild(tr);
  });
}

function appendLog(msg) {
  const div = document.createElement("div");
  div.className = "log-entry";
  div.innerText = msg;
  logStream.appendChild(div);
  logStream.scrollTop = logStream.scrollHeight;
}

// Mouse Selection (Smart Click + Drag ROI)
canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  // Check if clicking inside an auto-detected candidate box
  const clicked = currentDetections.find(det => {
    const [x, y, w, h] = det.bbox;
    return clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h;
  });

  if (clicked) {
    sendInitTarget(clicked.bbox, clicked.class_name);
    return;
  }

  isDrawingROI = true;
  startX = clickX; startY = clickY;
  currentX = clickX; currentY = clickY;
});

canvas.addEventListener("mousemove", (e) => {
  if (!isDrawingROI) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  currentX = (e.clientX - rect.left) * scaleX;
  currentY = (e.clientY - rect.top) * scaleY;
});

canvas.addEventListener("mouseup", () => {
  if (!isDrawingROI) return;
  isDrawingROI = false;
  const rx = Math.round(Math.min(startX, currentX));
  const ry = Math.round(Math.min(startY, currentY));
  const rw = Math.round(Math.abs(currentX - startX));
  const rh = Math.round(Math.abs(currentY - startY));

  if (rw > 15 && rh > 15) {
    // Custom drag box
    sendInitTarget([rx, ry, rw, rh], "Custom Target");
  } else {
    // Smart click fallback: creates a 70x70 bounding box around the click point
    const boxSize = 70;
    const bx = Math.max(0, Math.round(startX - boxSize / 2));
    const by = Math.max(0, Math.round(startY - boxSize / 2));
    sendInitTarget([bx, by, boxSize, boxSize], "Selected Object");
  }
});

function sendInitTarget(bbox, className) {
  const mode = document.getElementById("tracking-mode").value;
  graphDataHistory = [];
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      action: "init_roi",
      bbox: bbox,
      class_name: className,
      mode: mode
    }));
    appendLog(`Target selected: ${className} at [${bbox.join(", ")}]`);
  }
}

// Quick Target Fallback Button
document.getElementById("btn-quick-target").addEventListener("click", () => {
  // Center 80x80 box
  sendInitTarget([280, 140, 80, 80], "Center Target");
});

// Playback Controls
document.getElementById("btn-play").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "playback_action", sub_action: "play" }));
  }
});

document.getElementById("btn-pause").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "playback_action", sub_action: "pause" }));
  }
});

function triggerReset() {
  finishedBanner.style.display = "none";
  graphDataHistory = [];
  renderMovementGraph();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "playback_action", sub_action: "reset" }));
    appendLog("Video reset to frame 0. Ready for new target selection.");
  }
}

document.getElementById("btn-reset").addEventListener("click", triggerReset);
bannerBtnReset.addEventListener("click", triggerReset);

document.getElementById("speed-select").addEventListener("change", (e) => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "playback_action", sub_action: "set_speed", speed: parseFloat(e.target.value) }));
  }
});

// Sources & Uploads
document.getElementById("btn-webcam").addEventListener("click", async () => {
  await fetch(`${BACKEND_API}/use-webcam`, { method: "POST" });
  appendLog("Switched source to live Webcam.");
});

document.getElementById("video-upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  appendLog(`Uploading ${file.name}...`);
  const res = await fetch(`${BACKEND_API}/upload`, { method: "POST", body: formData });
  const data = await res.json();
  if (data.status === "success") appendLog(`Loaded recorded video: ${file.name}`);
});

// Exports
document.getElementById("btn-export-csv").addEventListener("click", () => {
  window.open(`${BACKEND_API}/export-data?format=csv`, "_blank");
});

bannerBtnExport.addEventListener("click", () => {
  window.open(`${BACKEND_API}/export-data?format=csv`, "_blank");
});

document.getElementById("btn-export-json").addEventListener("click", () => {
  window.open(`${BACKEND_API}/export-data?format=json`, "_blank");
});

document.getElementById("detection-algo").addEventListener("change", (e) => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "set_detection_algo", algo: e.target.value }));
  }
});

renderMovementGraph();
initWebSocket();