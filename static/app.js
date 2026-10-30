/* CCTV Surveillance System Complete Client Controller */

// Audio Alarm Synthesizer & Modal State
let sirenInterval = null;
let currentEmergencyAlertId = null;
let knownAlertIds = new Set();
let soundAlertsEnabled = true;

function playAlarmSound(severity) {
  if (!soundAlertsEnabled) return;
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);

    if (severity === "CRITICAL") {
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(920, audioCtx.currentTime);
      osc.frequency.setValueAtTime(680, audioCtx.currentTime + 0.18);
      osc.frequency.setValueAtTime(920, audioCtx.currentTime + 0.36);
      gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.55);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.55);
    } else {
      osc.type = "sine";
      osc.frequency.setValueAtTime(600, audioCtx.currentTime);
      gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.35);
    }
  } catch (e) {
    console.warn("Audio Context blocked: user interaction required.");
  }
}

function startSirenLoop(severity) {
  if (sirenInterval) return;
  playAlarmSound(severity);
  sirenInterval = setInterval(() => {
    playAlarmSound(severity);
  }, 600);
}

function stopSirenLoop() {
  if (sirenInterval) {
    clearInterval(sirenInterval);
    sirenInterval = null;
  }
}

window.dismissResultsModal = function() {
  stopSirenLoop();
  const modal = document.getElementById("resultsAlertModal");
  if (modal) {
    modal.classList.add("hidden");
    modal.classList.remove("active");
    modal.style.setProperty("display", "none", "important");
  }
};

window.acknowledgeEmergencyAlert = async function() {
  stopSirenLoop();
  const modal = document.getElementById("emergencyModal");
  if (modal) modal.classList.add("hidden");

  if (currentEmergencyAlertId) {
    await fetch(`/api/alerts/${currentEmergencyAlertId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "Acknowledged" })
    });
  }
};

window.testAlarmSound = function(sev) {
  playAlarmSound(sev);
};

window.toggleAudioAlerts = function() {
  soundAlertsEnabled = !soundAlertsEnabled;
  const btn = document.getElementById("soundToggleBtn");
  if (btn) {
    btn.textContent = soundAlertsEnabled ? "🔊 Sound Alarms: ON" : "🔇 Sound Alarms: OFF";
    btn.className = soundAlertsEnabled ? "button primary" : "button quiet";
  }
};

window.enablePushNotifications = function() {
  if ("Notification" in window) {
    Notification.requestPermission().then((permission) => {
      if (permission === "granted") {
        alert("Desktop notifications enabled!");
        new Notification("AI Surveillance", { body: "Desktop notifications active." });
      }
    });
  }
};

// Route controller
document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  if (page === "home") initSetupPage();
  else if (page === "roi") initRoiPage();
  else if (page === "live") initLivePage();
  else if (page === "alerts") initAlertsPage();
  else if (page === "results") initResultsPage();
});

/* ==============================================================
   1. SETUP & VIDEO UPLOAD
   ============================================================== */
function initSetupPage() {
  const fileInput = document.getElementById("videoFile");
  const uploadBtn = document.getElementById("uploadButton");
  const startBtn = document.getElementById("startAnalysis");
  const fileNameDisplay = document.getElementById("selectedFilename");
  const preview = document.getElementById("sourcePreview");
  const previewWrap = document.getElementById("previewWrap");
  const msgBox = document.getElementById("uploadMessage");

  let activeJobId = sessionStorage.getItem("cctv_active_job");
  let activeVideoUrl = sessionStorage.getItem("cctv_video_url");
  let activeFilename = sessionStorage.getItem("cctv_filename");

  if (activeJobId && activeVideoUrl) {
    preview.src = activeVideoUrl;
    previewWrap.classList.remove("hidden");
    startBtn.disabled = false;
    fileNameDisplay.textContent = activeFilename || "Video ready for analysis";
    document.getElementById("uploadState").textContent = "Video Ready";
    document.getElementById("uploadState").className = "badge success";
  }

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      fileNameDisplay.textContent = fileInput.files[0].name;
    }
  });

  uploadBtn.addEventListener("click", async () => {
    if (!fileInput.files.length) {
      alert("Please select a video file first!");
      return;
    }
    const formData = new FormData();
    formData.append("video", fileInput.files[0]);

    uploadBtn.disabled = true;
    uploadBtn.textContent = "Uploading...";
    msgBox.className = "notice hidden";

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");

      activeJobId = data.job_id;
      sessionStorage.setItem("cctv_active_job", data.job_id);
      sessionStorage.setItem("cctv_video_url", data.video_url);
      sessionStorage.setItem("cctv_filename", data.filename);

      preview.src = data.video_url;
      previewWrap.classList.remove("hidden");
      startBtn.disabled = false;
      document.getElementById("uploadState").textContent = "Video Ready";
      document.getElementById("uploadState").className = "badge success";
    } catch (err) {
      msgBox.textContent = err.message;
      msgBox.className = "notice error";
    } finally {
      uploadBtn.disabled = false;
      uploadBtn.textContent = "Upload video";
    }
  });

  startBtn.addEventListener("click", async () => {
    if (!activeJobId) {
      alert("Please upload a video first!");
      return;
    }

    const areaMode = document.getElementById("analysisArea").value;
    const savedRoi = JSON.parse(localStorage.getItem("cctv_saved_roi") || "[]");

    const config = {
      analysis_speed: document.getElementById("analysisSpeed").value,
      analysis_area: areaMode,
      extraction_mode: document.getElementById("extractionMode").value,
      confidence_threshold: parseFloat(document.getElementById("confidenceThreshold").value),
      polygons: (areaMode === "region") ? savedRoi : []
    };

    startBtn.disabled = true;
    document.getElementById("progressArea").classList.remove("hidden");

    try {
      const res = await fetch("/api/start-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: activeJobId, config })
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.error || "Failed to start analysis");
      }
      pollProgress(activeJobId);
    } catch (err) {
      alert(err.message);
      startBtn.disabled = false;
    }
  });

  function pollProgress(jobId) {
    const timer = setInterval(async () => {
      const res = await fetch("/api/status");
      const st = await res.json();
      if (st.status === "processing") {
        document.getElementById("progressPercent").textContent = `${Math.round(st.progress)}%`;
        document.getElementById("progressBar").style.width = `${st.progress}%`;
        document.getElementById("progressInfo").textContent = `Processing Frame: ${st.frame} / ${st.total_frames} (${st.processing_fps} FPS)`;
      } else if (st.status === "completed") {
        clearInterval(timer);
        window.location.href = `/results?job_id=${jobId}`;
      } else if (st.status === "error") {
        clearInterval(timer);
        alert(`Analysis Error: ${st.error}`);
        startBtn.disabled = false;
      }
    }, 1000);
  }
}

/* ==============================================================
   2. ROI POLYGON EDITOR
   ============================================================== */
function initRoiPage() {
  const canvas = document.getElementById("roiCanvas");
  const ctx = canvas.getContext("2d");
  const video = document.getElementById("roiVideo");
  const emptyNote = document.getElementById("roiEmpty");
  const saveBtn = document.getElementById("savePolygon");
  const clearBtn = document.getElementById("clearRoi");
  const undoBtn = document.getElementById("undoPoint");
  const playBtn = document.getElementById("roiPlayPause");
  const seek = document.getElementById("roiSeek");

  let points = [];
  const videoUrl = sessionStorage.getItem("cctv_video_url");

  if (videoUrl) {
    video.src = videoUrl;
    emptyNote.classList.add("hidden");
  }

  video.addEventListener("loadedmetadata", syncCanvas);
  video.addEventListener("play", syncCanvas);
  video.addEventListener("timeupdate", () => {
    if (video.duration) {
      seek.value = (video.currentTime / video.duration) * 100;
    }
  });

  seek.addEventListener("input", () => {
    if (video.duration) {
      video.currentTime = (seek.value / 100) * video.duration;
    }
  });

  playBtn.addEventListener("click", () => {
    if (video.paused) video.play();
    else video.pause();
  });

  function syncCanvas() {
    const rect = video.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      canvas.width = rect.width;
      canvas.height = rect.height;
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      draw();
    }
  }

  window.addEventListener("resize", syncCanvas);

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / canvas.width;
    const ny = (e.clientY - rect.top) / canvas.height;
    points.push([Math.max(0, Math.min(1, nx)), Math.max(0, Math.min(1, ny))]);
    draw();
  });

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!points.length) return;

    ctx.strokeStyle = "#3b82f6";
    ctx.fillStyle = "rgba(59, 130, 246, 0.25)";
    ctx.lineWidth = 2;

    ctx.beginPath();
    points.forEach((p, idx) => {
      const x = p[0] * canvas.width;
      const y = p[1] * canvas.height;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      ctx.fillStyle = "#ef4444";
      ctx.fillRect(x - 4, y - 4, 8, 8);
      ctx.fillStyle = "rgba(59, 130, 246, 0.25)";
    });

    if (points.length >= 3) {
      ctx.closePath();
      ctx.fill();
    }
    ctx.stroke();
  }

  undoBtn.addEventListener("click", () => { points.pop(); draw(); });
  clearBtn.addEventListener("click", () => { 
    points = []; 
    draw(); 
    localStorage.removeItem("cctv_saved_roi"); 
    alert("Saved ROI cleared.");
  });

  saveBtn.addEventListener("click", () => {
    if (points.length < 3) {
      alert("Click at least 3 points on the video to outline a polygon.");
      return;
    }
    localStorage.setItem("cctv_saved_roi", JSON.stringify([points]));
    alert("ROI Polygon successfully saved! Redirecting to Setup...");
    window.location.href = "/";
  });

  const saved = localStorage.getItem("cctv_saved_roi");
  if (saved) {
    try {
      points = JSON.parse(saved)[0] || [];
      setTimeout(syncCanvas, 200);
    } catch (e) {}
  }
}

/* ==============================================================
   3. LIVE STREAM MONITORING (Button & Stream Controller)
   ============================================================== */
function initLivePage() {
  const form = document.getElementById("liveForm");
  const stopBtn = document.getElementById("stopLive");
  const startBtn = document.getElementById("startLiveBtn");
  const feed = document.getElementById("liveFeed");
  const placeholder = document.getElementById("feedPlaceholder");
  const errBanner = document.getElementById("liveErrorBanner");

  let pollInterval = null;

  async function handleStartStream(e) {
    if (e) e.preventDefault();
    const urlInput = document.getElementById("streamUrl");
    const nameInput = document.getElementById("sourceName");

    if (!urlInput || !urlInput.value.trim()) {
      alert("Please enter a stream URL (or 0 for webcam).");
      return;
    }

    const url = urlInput.value.trim();
    const name = nameInput ? nameInput.value.trim() : "Live Camera";

    console.log("[CCTV Live] Starting stream connection to:", url);
    errBanner.classList.add("hidden");
    startBtn.disabled = true;
    startBtn.textContent = "Connecting...";

    try {
      const res = await fetch("/api/live/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stream_url: url,
          source_name: name,
          polygons: JSON.parse(localStorage.getItem("cctv_saved_roi") || "[]")
        })
      });
      const d = await res.json();
      if (!res.ok) throw new Error(d.error || "Cannot connect to stream");

      // Set image source and unhide video element
      setTimeout(() => {
        feed.src = "/api/live/feed?" + Date.now();
        feed.classList.remove("hidden");
        feed.style.display = "block";
        placeholder.classList.add("hidden");
        placeholder.style.display = "none";
        stopBtn.disabled = false;
        startBtn.textContent = "Monitoring Active";
      }, 350);

      if (pollInterval) clearInterval(pollInterval);
      pollInterval = setInterval(updateLiveMetrics, 1000);
    } catch (err) {
      console.error("[CCTV Live Error]:", err);
      errBanner.textContent = err.message;
      errBanner.classList.remove("hidden");
      startBtn.disabled = false;
      startBtn.textContent = "Start Monitoring";
    }
  }

  // Attach both form submit and direct button click so it never misses
  if (form) form.addEventListener("submit", handleStartStream);
  if (startBtn) startBtn.addEventListener("click", handleStartStream);

  stopBtn.addEventListener("click", async () => {
    console.log("[CCTV Live] Stopping stream...");
    await fetch("/api/live/stop", { method: "POST" });
    feed.src = "";
    feed.classList.add("hidden");
    feed.style.display = "none";
    placeholder.classList.remove("hidden");
    placeholder.style.display = "block";
    stopBtn.disabled = true;
    startBtn.disabled = false;
    startBtn.textContent = "Start Monitoring";
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  });

  async function updateLiveMetrics() {
    try {
      const res = await fetch("/api/live/status");
      const st = await res.json();

      if (st.status === "error") {
        errBanner.textContent = st.error || "Camera connection lost.";
        errBanner.classList.remove("hidden");
        feed.classList.add("hidden");
        feed.style.display = "none";
        placeholder.classList.remove("hidden");
        placeholder.style.display = "block";
        stopBtn.disabled = true;
        startBtn.disabled = false;
        startBtn.textContent = "Start Monitoring";
        if (pollInterval) {
          clearInterval(pollInterval);
          pollInterval = null;
        }
        return;
      }

      document.getElementById("sourceFps").textContent = st.source_fps || "—";
      document.getElementById("processingFps").textContent = st.processing_fps || "—";
      document.getElementById("detectionCount").textContent = st.current_detections || "0";
    } catch (e) {}
  }
}

/* ==============================================================
   4. EMERGENCY ALERTS DASHBOARD
   ============================================================== */
async function initAlertsPage() {
  const listEl = document.getElementById("alertList");
  const countEl = document.getElementById("alertCount");

  window.loadAlerts = async function() {
    try {
      const res = await fetch("/api/alerts");
      const d = await res.json();
      listEl.innerHTML = "";

      if (!d.alerts || !d.alerts.length) {
        listEl.innerHTML = `<p class="muted">No emergency alerts recorded yet.</p>`;
        if (countEl) countEl.textContent = "0";
        return;
      }

      if (countEl) countEl.textContent = d.alerts.length;

      d.alerts.forEach((alert) => {
        const card = document.createElement("div");
        card.className = "card";
        card.style.marginBottom = "14px";
        card.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <strong style="font-size:14px;">${alert.message}</strong>
            <span class="badge ${alert.severity.toLowerCase()}">${alert.severity}</span>
          </div>
          <div style="color:var(--text-muted); font-size:11px; font-family:var(--font-mono); margin-bottom:12px;">
            Source: ${alert.source} | Time: ${alert.created_at} | Status: <strong>${alert.status}</strong>
          </div>
          ${
            alert.snapshot_url
              ? `<div style="margin-bottom:12px;">
                   <img src="${alert.snapshot_url}" 
                        style="max-width:380px; width:100%; border-radius:6px; display:block; background:#000; border:1px solid var(--border);" 
                        alt="Evidence Snapshot"
                        onerror="this.onerror=null; this.parentElement.innerHTML='<span style=\\'color:#ef4444; font-size:11px;\\'>Snapshot recording in progress...</span>'">
                 </div>`
              : ""
          }
          <div style="display:flex; gap:8px;">
            <button class="button quiet" onclick="updateAlertStatus('${alert.id}', 'Acknowledged')">Acknowledge</button>
            <button class="button quiet" onclick="updateAlertStatus('${alert.id}', 'Resolved')">Resolve</button>
          </div>
        `;
        listEl.appendChild(card);
      });
    } catch (err) {
      console.error("Error loading alerts:", err);
    }
  };

  window.updateAlertStatus = async (id, status) => {
    await fetch(`/api/alerts/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status })
    });
    window.loadAlerts();
  };

  window.loadAlerts();
}

/* ==============================================================
   5. RESULTS PAGE
   ============================================================== */
async function initResultsPage() {
  const urlParams = new URLSearchParams(window.location.search);
  const jobId = urlParams.get("job_id");
  if (!jobId) return;

  const res = await fetch(`/api/results/${jobId}/summary`);
  if (!res.ok) return;
  const summary = await res.json();

  document.getElementById("resultFilename").textContent = summary.video;
  document.getElementById("resultState").textContent = "Completed";
  document.getElementById("resultState").className = "badge success";

  if (summary.alerts && summary.alerts.length > 0) {
    triggerResultsAlertPopup(summary.alerts);
  } else {
    fetch(`/api/alerts?source=${encodeURIComponent(summary.video)}&status=New`)
      .then(r => r.json())
      .then(d => {
        if (d.alerts && d.alerts.length > 0) {
          triggerResultsAlertPopup(d.alerts);
        }
      })
      .catch(() => {});
  }

  const procVid = document.getElementById("processedVideo");
  const procPanel = document.getElementById("processedPanel");
  procVid.src = `/api/results/${jobId}/files/processed_video.mp4`;
  procPanel.classList.remove("hidden");
  procPanel.style.display = "block";

  const clipsGrid = document.getElementById("eventClipsGrid");
  const clipsPanel = document.getElementById("eventVideosPanel");
  if (summary.event_clips && summary.event_clips.length > 0) {
    clipsGrid.innerHTML = "";
    summary.event_clips.forEach((clipPath) => {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `
        <video src="/api/results/${jobId}/files/${clipPath}" controls style="width:100%; border-radius:4px; margin-bottom:6px;"></video>
        <span style="font-size:11px; font-family:var(--font-mono); color:var(--text-muted);">${clipPath.split("/").pop()}</span>
      `;
      clipsGrid.appendChild(card);
    });
    clipsPanel.classList.remove("hidden");
    clipsPanel.style.display = "block";
  }

  const framesGrid = document.getElementById("eventFramesGrid");
  const framesPanel = document.getElementById("eventFramesPanel");
  if (summary.event_frames && summary.event_frames.length > 0) {
    framesGrid.innerHTML = "";
    summary.event_frames.forEach((framePath) => {
      const img = document.createElement("img");
      img.src = `/api/results/${jobId}/files/${framePath}`;
      img.style.width = "100%";
      img.style.borderRadius = "4px";
      framesGrid.appendChild(img);
    });
    framesPanel.classList.remove("hidden");
    framesPanel.style.display = "block";
  }

  const graphsGrid = document.getElementById("graphsGrid");
  const graphsPanel = document.getElementById("graphsPanel");
  if (summary.graphs && summary.graphs.length > 0) {
    graphsGrid.innerHTML = "";
    summary.graphs.forEach((gPath) => {
      const img = document.createElement("img");
      img.src = `/api/results/${jobId}/files/${gPath}`;
      img.style.maxWidth = "360px";
      img.style.borderRadius = "6px";
      img.style.border = "1px solid var(--border)";
      graphsGrid.appendChild(img);
    });
    graphsPanel.classList.remove("hidden");
    graphsPanel.style.display = "block";
  }

  const csvContainer = document.getElementById("csvLinks");
  const datasetPanel = document.getElementById("datasetPanel");
  if (summary.datasets && summary.datasets.length > 0) {
    csvContainer.innerHTML = "";
    summary.datasets.forEach((csvFile) => {
      const link = document.createElement("a");
      link.className = "button quiet";
      link.href = `/api/results/${jobId}/files/${csvFile}?download=1`;
      link.textContent = `Download ${csvFile.split("/").pop()}`;
      csvContainer.appendChild(link);
    });
    datasetPanel.classList.remove("hidden");
    datasetPanel.style.display = "block";
  }

  const dlBtn = document.getElementById("downloadAll");
  dlBtn.href = `/api/results/${jobId}/download`;
  dlBtn.classList.remove("hidden");
}

function triggerResultsAlertPopup(alerts) {
  const modal = document.getElementById("resultsAlertModal");
  const banner = document.getElementById("hazardAlertBanner");
  if (!alerts || !alerts.length) return;

  const topAlert = alerts[0];
  const isCritical = alerts.some(a => a.severity === "CRITICAL");
  const sev = isCritical ? "CRITICAL" : (topAlert.severity || "HIGH");

  if (banner) {
    banner.classList.remove("hidden");
    document.getElementById("hazardBannerText").textContent =
      `🚨 Attention: ${alerts.length} danger incident(s) detected during video processing!`;
  }

  if (modal) {
    document.getElementById("modalAlertTitle").textContent = `${sev} HAZARD DETECTED IN FOOTAGE`;
    document.getElementById("modalAlertMessage").textContent = topAlert.message;
    document.getElementById("modalAlertMeta").textContent =
      `Source: ${topAlert.source} | Status: New Incident`;

    const img = document.getElementById("modalAlertSnapshot");
    if (topAlert.snapshot_url) {
      img.src = topAlert.snapshot_url;
      img.style.display = "block";
    } else {
      img.style.display = "none";
    }

    modal.classList.remove("hidden");
    modal.classList.add("active");
    modal.style.setProperty("display", "flex", "important");

    startSirenLoop(sev);
  }
}