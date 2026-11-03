/**
 * SERBot BuildSentinel Frontend Controller
 * Handles live stream views, image inspection, telemetry updates, and UI reactivity.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const btnStream = document.getElementById("mode-stream");
  const btnInspect = document.getElementById("mode-inspect");
  const streamView = document.getElementById("stream-view");
  const inspectView = document.getElementById("inspect-view");
  const uploadOverlay = document.getElementById("upload-overlay");
  const fileInput = document.getElementById("file-input");
  const sampleBar = document.getElementById("sample-bar");
  const liveTag = document.getElementById("live-tag");
  const btnExport = document.getElementById("btn-export-json");

  // Telemetry Elements
  const telemGas = document.getElementById("telem-gas");
  const telemTemp = document.getElementById("telem-temp");
  const telemDist = document.getElementById("telem-dist");
  const telemVib = document.getElementById("telem-vib");
  const telemBat = document.getElementById("telem-bat");

  // Metrics Elements
  const metaLatency = document.getElementById("meta-latency");
  const metaFps = document.getElementById("meta-fps");
  const scoreBanner = document.getElementById("score-banner");
  const overallScore = document.getElementById("overall-score");
  const overallStatus = document.getElementById("overall-status");
  const navCommand = document.getElementById("nav-command");
  const navStatus = document.getElementById("nav-status");

  // PPE Elements
  const ppeBadge = document.getElementById("ppe-compliance-badge");
  const workersCount = document.getElementById("workers-count");
  const ppeViolationsCount = document.getElementById("ppe-violations-count");
  const workerReportsList = document.getElementById("worker-reports-list");

  // Structural Elements
  const structuralBadge = document.getElementById("structural-badge");
  const cracksCount = document.getElementById("cracks-count");
  const structDisp = document.getElementById("structural-displacement");
  const structVec = document.getElementById("structural-vector");

  // Hazards Elements
  const hazardsBadge = document.getElementById("hazards-badge");
  const fireCount = document.getElementById("fire-count");
  const spillsCount = document.getElementById("spills-count");
  const zoneBreachesCount = document.getElementById("zone-breaches-count");

  // Alerts Feed
  const alertsFeed = document.getElementById("alerts-feed");
  const alertsTotalBadge = document.getElementById("alerts-total-badge");

  let activeReport = null;
  let currentMode = "stream"; // 'stream' or 'inspect'

  // Initialize Sample List
  loadSamples();

  // Mode Switchers
  btnStream.addEventListener("click", () => {
    currentMode = "stream";
    btnStream.classList.add("active");
    btnInspect.classList.remove("active");
    streamView.style.display = "block";
    inspectView.style.display = "none";
    uploadOverlay.style.display = "none";
    liveTag.style.display = "flex";
  });

  btnInspect.addEventListener("click", () => {
    currentMode = "inspect";
    btnInspect.classList.add("active");
    btnStream.classList.remove("active");
    streamView.style.display = "none";
    liveTag.style.display = "none";

    if (!inspectView.src || inspectView.src.length < 50) {
      uploadOverlay.style.display = "flex";
      inspectView.style.display = "none";
    } else {
      uploadOverlay.style.display = "none";
      inspectView.style.display = "block";
    }
  });

  // File Upload Handlers
  uploadOverlay.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      uploadImageFile(e.target.files[0]);
    }
  });

  // Drag and Drop
  uploadOverlay.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadOverlay.style.borderColor = "var(--accent-cyan)";
  });
  uploadOverlay.addEventListener("dragleave", () => {
    uploadOverlay.style.borderColor = "rgba(0, 242, 254, 0.4)";
  });
  uploadOverlay.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      uploadImageFile(e.dataTransfer.files[0]);
    }
  });

  // Export JSON Report
  btnExport.addEventListener("click", () => {
    if (!activeReport) {
      alert("No active inspection report to export yet. Please run an image analysis or sample test!");
      return;
    }
    const blob = new Blob([JSON.stringify(activeReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `BuildSentinel_Safety_Audit_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  // Fetch Sample Chips
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      sampleBar.innerHTML = "";
      
      data.samples.forEach(sample => {
        const chip = document.createElement("button");
        chip.className = "sample-chip";
        chip.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <polyline points="21 15 16 10 5 21"/>
          </svg>
          ${sample.name}
        `;
        chip.addEventListener("click", () => runSampleAnalysis(sample.id, chip));
        sampleBar.appendChild(chip);
      });
    } catch (e) {
      console.error("Failed to load sample chips:", e);
    }
  }

  // Trigger Sample Analysis
  async function runSampleAnalysis(sampleId, chipElement) {
    // Switch to inspect mode
    btnInspect.click();
    
    // Highlight active chip
    document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
    if (chipElement) chipElement.classList.add("active");

    uploadOverlay.style.display = "none";
    metaLatency.textContent = "Processing...";
    
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: sampleId })
      });
      const report = await res.json();
      updateUI(report);
    } catch (e) {
      console.error("Error analyzing sample:", e);
      alert("Error analyzing sample image.");
    }
  }

  // Upload custom user image
  async function uploadImageFile(file) {
    btnInspect.click();
    uploadOverlay.style.display = "none";
    metaLatency.textContent = "Uploading & Analyzing...";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });
      const report = await res.json();
      updateUI(report);
    } catch (e) {
      console.error("Error analyzing uploaded file:", e);
      alert("Failed to process image file.");
    }
  }

  // Update All UI Components
  function updateUI(report) {
    activeReport = report;

    // Viewport image
    if (report.annotated_image_b64) {
      inspectView.src = report.annotated_image_b64;
      inspectView.style.display = "block";
      uploadOverlay.style.display = "none";
    }

    // Performance meta
    metaLatency.textContent = `${report.latency_ms} ms`;
    metaFps.textContent = `${report.fps}`;

    // Sensor Fusion Telemetry
    const telem = report.fusion?.telemetry || {};
    telemGas.innerHTML = `${telem.gas_smoke_ppm ?? '--'} <small style="font-size: 11px; color: var(--text-dim);">ppm</small>`;
    telemTemp.innerHTML = `${telem.temperature_c ?? '--'} <small style="font-size: 11px; color: var(--text-dim);">°C</small>`;
    telemDist.innerHTML = `${telem.ultrasonic_distance_cm ?? '--'} <small style="font-size: 11px; color: var(--text-dim);">cm</small>`;
    telemVib.innerHTML = `${telem.vibration_g ?? '--'} <small style="font-size: 11px; color: var(--text-dim);">g</small>`;
    telemBat.textContent = `${telem.battery_pct ?? '--'}%`;

    // Overall Safety Score & Status
    const score = report.fusion?.site_safety_score ?? 100;
    const status = report.fusion?.overall_status ?? "OPTIMAL_SAFE";
    overallScore.textContent = score;
    overallStatus.textContent = status;

    if (score < 50 || status.includes("CRITICAL")) {
      scoreBanner.className = "score-banner hazard";
    } else {
      scoreBanner.className = "score-banner";
    }

    // Robotic Navigation Directive
    const nav = report.obstacles?.nav_command ?? "PROCEED_SAFE";
    navCommand.textContent = nav;
    navCommand.className = `badge ${nav.includes("STOP") ? "critical" : (nav.includes("STEER") ? "warning" : "safe")}`;
    navStatus.textContent = report.obstacles?.nav_status ?? "Normal operation";

    // PPE Compliance & Workers
    const ppe = report.ppe || {};
    const compRate = ppe.compliance_rate ?? 100.0;
    const workers = ppe.worker_reports || [];
    const ppeViolations = ppe.violations || [];

    workersCount.textContent = workers.length;
    ppeViolationsCount.textContent = ppeViolations.length;
    ppeViolationsCount.className = `badge ${ppeViolations.length > 0 ? "critical" : "safe"}`;
    ppeBadge.textContent = `${compRate}% COMPLIANT`;
    ppeBadge.className = `badge ${compRate >= 90 ? "safe" : (compRate >= 50 ? "warning" : "critical")}`;

    // Render individual worker status rows
    workerReportsList.innerHTML = "";
    if (workers.length === 0) {
      workerReportsList.innerHTML = `<div style="font-size: 12px; color: var(--text-dim);">No workers detected in current sector.</div>`;
    } else {
      workers.forEach(w => {
        const row = document.createElement("div");
        row.className = "metric-row";
        row.innerHTML = `
          <span>Worker #${w.worker_id}</span>
          <span class="badge ${w.is_compliant ? 'safe' : 'critical'}">
            ${w.is_compliant ? 'Compliant' : 'Missing: ' + w.missing_gear.join(', ')}
          </span>
        `;
        workerReportsList.appendChild(row);
      });
    }

    // Structural Health
    const structural = report.structural || {};
    const cracks = structural.cracks || [];
    const movement = structural.movement || {};

    cracksCount.textContent = cracks.length;
    structDisp.textContent = `${movement.displacement_px ?? 0.0} px`;
    structVec.textContent = JSON.stringify(movement.vector ?? [0.0, 0.0]);

    if (cracks.length > 0 || movement.movement_detected) {
      structuralBadge.textContent = "DEFECT / SHIFT";
      structuralBadge.className = "badge critical";
    } else {
      structuralBadge.textContent = "NORMAL";
      structuralBadge.className = "badge safe";
    }

    // Environmental Hazards & Zones
    const hazards = report.hazards?.hazards || [];
    const zoneBreaches = report.zones?.total_breaches || 0;

    const fireThreats = hazards.filter(h => h.class.includes("fire") || h.class.includes("flame"));
    const spillThreats = hazards.filter(h => h.class.includes("spill"));

    fireCount.textContent = fireThreats.length > 0 ? `${fireThreats.length} DETECTED` : "None";
    fireCount.className = `badge ${fireThreats.length > 0 ? "critical" : "safe"}`;

    spillsCount.textContent = spillThreats.length > 0 ? `${spillThreats.length} DETECTED` : "None";
    spillsCount.className = `badge ${spillThreats.length > 0 ? "warning" : "safe"}`;

    zoneBreachesCount.textContent = zoneBreaches;
    zoneBreachesCount.className = `badge ${zoneBreaches > 0 ? "critical" : "safe"}`;

    if (fireThreats.length > 0 || zoneBreaches > 0) {
      hazardsBadge.textContent = "ACTIVE HAZARD";
      hazardsBadge.className = "badge critical";
    } else if (spillThreats.length > 0) {
      hazardsBadge.textContent = "CAUTION";
      hazardsBadge.className = "badge warning";
    } else {
      hazardsBadge.textContent = "CLEAR";
      hazardsBadge.className = "badge safe";
    }

    // Live Alert Feed
    const allAlerts = report.all_alerts || [];
    alertsTotalBadge.textContent = `${allAlerts.length} Alerts`;
    alertsFeed.innerHTML = "";

    if (allAlerts.length === 0) {
      alertsFeed.innerHTML = `<div style="font-size: 12px; color: var(--text-dim); text-align: center; padding: 20px 0;">No active hazards or safety breaches detected.</div>`;
    } else {
      allAlerts.forEach(alert => {
        const sev = alert.severity || "INFO";
        const item = document.createElement("div");
        item.className = `alert-item ${sev}`;
        item.innerHTML = `
          <div class="alert-title">
            <span>${alert.type || 'HAZARD'}</span>
            <span class="badge ${sev.toLowerCase()}">${sev}</span>
          </div>
          <div class="alert-msg">${alert.message}</div>
        `;
        alertsFeed.appendChild(item);
      });
    }
  }

});
