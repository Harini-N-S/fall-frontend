/**
 * Nurse Fall Alert & Escalation Dashboard Client
 * Real-time WebSocket connection, live countdowns, state rendering, audio alerts.
 */

// Global Dashboard State
let appState = {
  summary: {
    active_critical: 0,
    awaiting_response: 0,
    being_attended: 0,
    resolved_today: 0,
    timeout_seconds: 30,
  },
  active_alerts: [],
  resolved_alerts: [],
  care_team: [],
  timeout_seconds: 30,
};

let ws = null;
let soundEnabled = true;
let audioCtx = null;
let auditTrailTargetAlert = null;

// Initialize when DOM loads
document.addEventListener("DOMContentLoaded", () => {
  setupClock();
  setupEventListeners();
  connectWebSocket();
  fetchInitialState();
});

// Real-time Clock
function setupClock() {
  const clockEl = document.getElementById("header-clock");
  function updateTime() {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    });
  }
  updateTime();
  setInterval(updateTime, 1000);
}

// WebSocket Management
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[WS] Connected to Nurse Dashboard Server.");
    updateSystemOnlineBadge(true);
  };

  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      handleServerEvent(payload);
    } catch (e) {
      console.error("[WS] Error parsing message:", e);
    }
  };

  ws.onclose = () => {
    console.warn("[WS] Disconnected. Reconnecting in 2 seconds...");
    updateSystemOnlineBadge(false);
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error("[WS] WebSocket error:", err);
  };
}

function updateSystemOnlineBadge(isOnline) {
  const badge = document.getElementById("system-status-badge");
  if (!badge) return;
  if (isOnline) {
    badge.innerHTML = `<span class="live-pulse"></span> ONLINE / REAL-TIME`;
    badge.style.color = "#34d399";
  } else {
    badge.innerHTML = `<span class="summary-dot dot-critical"></span> DISCONNECTED (RETRYING)`;
    badge.style.color = "#f87171";
  }
}

// Event Dispatcher
function handleServerEvent(payload) {
  const eventType = payload.event;

  if (eventType === "INITIAL_STATE") {
    appState = payload.data;
    renderFullDashboard();
    return;
  }

  if (eventType === "TIMER_TICK") {
    // Synchronize countdowns
    if (payload.data && payload.data.alerts) {
      updateTimersOnly(payload.data.alerts);
    }
    return;
  }

  // State updates requiring refresh
  if (payload.summary) {
    appState.summary = payload.summary;
    renderSummaryCards();
  }

  if (eventType === "ALERT_CREATED") {
    playTelemetryChime("critical");
    fetchInitialState();
  } else if (eventType === "ALERT_ESCALATED") {
    playTelemetryChime("escalation");
    fetchInitialState();
  } else if (eventType === "ALERT_RESPONDED" || eventType === "ALERT_CARE_IN_PROGRESS") {
    playTelemetryChime("attended");
    fetchInitialState();
  } else if (eventType === "ALERT_RESOLVED") {
    fetchInitialState();
  } else if (eventType === "NURSE_STATUS_CHANGED" || eventType === "DEMO_RESET") {
    fetchInitialState();
  }
}

// Fetch Full State from REST
async function fetchInitialState() {
  try {
    const res = await fetch("/api/state");
    if (res.ok) {
      appState = await res.json();
      renderFullDashboard();
    }
  } catch (e) {
    console.error("Failed to fetch dashboard state:", e);
  }
}

// Render Complete UI
function renderFullDashboard() {
  renderSummaryCards();
  renderProminentEmergencyBanner();
  renderActiveAlertsTable();
  renderCareTeamList();
  renderResolvedAlertsTable();

  // Update timeout select value
  const timeoutSelect = document.getElementById("timeout-select");
  if (timeoutSelect && appState.timeout_seconds) {
    timeoutSelect.value = String(appState.timeout_seconds);
  }
}

// 1. Top Summary Cards
function renderSummaryCards() {
  const s = appState.summary || {};
  document.getElementById("count-active-critical").textContent = s.active_critical ?? 0;
  document.getElementById("count-awaiting-response").textContent = s.awaiting_response ?? 0;
  document.getElementById("count-being-attended").textContent = s.being_attended ?? 0;
  document.getElementById("count-resolved-today").textContent = s.resolved_today ?? 0;
  document.getElementById("active-alerts-count-badge").textContent = `${appState.active_alerts?.length || 0} active`;
  document.getElementById("resolved-count-badge").textContent = `${appState.resolved_alerts?.length || 0} resolved`;
}

// 2. Prominent Emergency Alert Showcase (Section 4 & 6)
function renderProminentEmergencyBanner() {
  const container = document.getElementById("emergency-banner-container");
  if (!container) return;

  // Find the most critical alert waiting for nurse or doctor response
  const activeAlerts = appState.active_alerts || [];
  const waitingAlert = activeAlerts.find(
    (a) => a.state === "WAITING_FOR_RESPONSE" || a.state === "DOCTOR_ESCALATION" || a.state === "ESCALATING"
  );

  if (!waitingAlert) {
    container.innerHTML = "";
    return;
  }

  const isDoctor = waitingAlert.state === "DOCTOR_ESCALATION";
  const timerSec = waitingAlert.timer_seconds_remaining ?? 30;
  const timerStr = timerSec < 10 ? `0${timerSec}` : `${timerSec}`;

  let trailHtml = "";
  if (waitingAlert.escalation_history && waitingAlert.escalation_history.length > 0) {
    trailHtml = waitingAlert.escalation_history
      .map((h) => {
        let cls = "trail-node";
        if (h.event_type === "NO_RESPONSE") cls += " timeout";
        if (h.event_type === "ESCALATED") cls += " active-nurse";
        if (h.event_type === "DOCTOR_ESCALATION") cls += " timeout";
        return `<span class="${cls}">${h.time_str} — ${escapeHtml(h.actor)} (${escapeHtml(h.event_type)})</span>`;
      })
      .join('<span class="trail-arrow">→</span>');
  }

  const doctorReasonNotice = isDoctor
    ? `<div style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.4); padding: 8px 12px; border-radius: 6px; font-size: 0.8rem; color: #f472b6; margin-bottom: 12px;">
        <strong>REASON:</strong> No nurse acknowledged fall alert after full escalation chain. Immediate Doctor Care Required!
       </div>`
    : "";

  container.innerHTML = `
    <div class="alert-card-prominent ${isDoctor ? "alert-card-doctor" : ""}" id="prominent-card-${waitingAlert.alert_id}">
      <div class="emergency-header">
        <div class="emergency-title-box">
          <div class="emergency-badge ${isDoctor ? "badge-doctor" : ""}">
            ${isDoctor ? "🩺 DOCTOR ESCALATION" : "🔴 CRITICAL FALL ALERT"}
          </div>
          <div class="emergency-status-text">
            ${isDoctor ? "DOCTOR RESPONSE REQUIRED" : "WAITING FOR NURSE RESPONSE"}
          </div>
        </div>
        <div style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-muted);">
          ALERT ID: <strong>${waitingAlert.alert_id}</strong>
        </div>
      </div>

      ${doctorReasonNotice}

      <div class="emergency-content-grid">
        <div class="patient-info-block">
          <div class="patient-name-large">${escapeHtml(waitingAlert.patient_id)}</div>
          <div class="room-bed-pill">
            <span>📍 ${escapeHtml(waitingAlert.room_bed)}</span>
          </div>
          <div class="meta-detected-time">
            Detected: <strong>${waitingAlert.detected_time_str}</strong>
          </div>
        </div>

        <div class="nurse-alerting-block">
          <div class="alerting-label">${isDoctor ? "NOW ALERTING PHYSICIAN" : "ASSIGNED / NOW ALERTING"}</div>
          <div class="alerting-nurse-name">
            ${isDoctor ? "🩺 " : "👩‍⚕️ "}${escapeHtml(waitingAlert.assigned_person_name || "Primary Nurse")}
          </div>
          <div class="escalation-step-badge">
            ${isDoctor ? "Doctor Level Escalation" : `Escalation Step ${waitingAlert.escalation_index + 1} of 3`}
          </div>
          <div style="font-size: 0.75rem; color: var(--text-dim); margin-top: 6px;">
            Sequential Escalation Window: 30s
          </div>
        </div>

        <div class="timer-action-block">
          <div class="timer-box">
            <span class="timer-label">RESPONSE COUNTDOWN</span>
            <span class="timer-countdown-big ${isDoctor ? "timer-countdown-doctor" : ""}" id="prominent-countdown">
              00:${timerStr}
            </span>
          </div>
          <button class="btn-respond-large ${isDoctor ? "btn-doctor-respond" : ""}" onclick="respondAlert('${waitingAlert.alert_id}', '${escapeHtml(waitingAlert.assigned_person_name)}')">
            ⚡ RESPOND
          </button>
        </div>
      </div>

      <div class="card-escalation-trail">
        <div class="trail-title">Escalation Sequence & Audit Trail:</div>
        <div class="trail-nodes">
          <span class="trail-node">${waitingAlert.detected_time_str} — Fall Detected</span>
          <span class="trail-arrow">→</span>
          ${trailHtml}
        </div>
      </div>
    </div>
  `;
}

// 3. Active Alert Table (Section 3)
function renderActiveAlertsTable() {
  const tbody = document.getElementById("active-alerts-tbody");
  if (!tbody) return;

  const alerts = appState.active_alerts || [];
  if (alerts.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="9" class="empty-placeholder">
          🟢 No active fall alerts. All patients stable and monitored.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = alerts
    .map((alert) => {
      const isDoctor = alert.state === "DOCTOR_ESCALATION";
      const isWaiting = alert.state === "WAITING_FOR_RESPONSE";
      const isAttended =
        alert.state === "ACKNOWLEDGED" ||
        alert.state === "CARE_IN_PROGRESS" ||
        alert.state === "DOCTOR_ACKNOWLEDGED";

      let rowClass = "";
      if (isDoctor) rowClass = "tr-doctor";
      else if (isWaiting) rowClass = "tr-waiting";
      else if (isAttended) rowClass = "tr-attended";

      let statusBadge = "";
      let actionButtons = "";
      let timerCell = "";

      if (isWaiting) {
        statusBadge = `<span class="status-badge status-waiting">⏳ WAITING FOR RESPONSE</span>`;
        const sec = alert.timer_seconds_remaining ?? 30;
        const formatted = sec < 10 ? `00:0${sec}` : `00:${sec}`;
        timerCell = `<span class="timer-cell active-warning" id="tbl-timer-${alert.alert_id}">${formatted}</span>`;
        actionButtons = `
          <button class="btn-action-respond" onclick="respondAlert('${alert.alert_id}', '${escapeHtml(alert.assigned_person_name)}')">
            RESPOND
          </button>
        `;
      } else if (isDoctor) {
        statusBadge = `<span class="status-badge status-doctor">🚨 DOCTOR ESCALATION</span>`;
        const sec = alert.timer_seconds_remaining ?? 30;
        const formatted = sec < 10 ? `00:0${sec}` : `00:${sec}`;
        timerCell = `<span class="timer-cell active-warning" id="tbl-timer-${alert.alert_id}">${formatted}</span>`;
        actionButtons = `
          <button class="btn-action-respond btn-doctor-respond" onclick="respondAlert('${alert.alert_id}', '${escapeHtml(alert.assigned_person_name)}')">
            DOCTOR RESPOND
          </button>
        `;
      } else if (isAttended) {
        const nurseName = alert.responding_nurse || alert.assigned_person_name || "Nurse";
        statusBadge = `
          <span class="status-badge status-attended">
            🟢 ATTENDED by ${escapeHtml(nurseName)}
          </span>
        `;
        timerCell = `<span class="timer-cell stopped">STOPPED</span>`;
        actionButtons = `
          <button class="btn-action-resolve" onclick="resolveAlert('${alert.alert_id}', '${escapeHtml(nurseName)}')">
            RESOLVE / CARE COMPLETED
          </button>
        `;
      }

      // Escalation subtext
      let assignedSub = "";
      if (alert.escalation_index > 0) {
        assignedSub = `<span class="assigned-escalated-sub">Escalated (Level ${alert.escalation_index + 1})</span>`;
      }

      return `
        <tr class="${rowClass}" id="row-${alert.alert_id}">
          <td>
            <span class="priority-badge priority-critical">CRITICAL</span>
          </td>
          <td class="patient-cell">${escapeHtml(alert.patient_id)}</td>
          <td class="room-cell">${escapeHtml(alert.room_bed)}</td>
          <td style="font-weight: 600; color: #f87171;">FALL DETECTED</td>
          <td style="font-family: var(--font-mono); font-size: 0.8rem;">${alert.detected_time_str}</td>
          <td class="assigned-cell">
            ${escapeHtml(alert.assigned_person_name || "Assigned Nurse")}
            ${assignedSub}
          </td>
          <td>${statusBadge}</td>
          <td>${timerCell}</td>
          <td>
            <div style="display: flex; gap: 6px;">
              ${actionButtons}
              <button class="btn-view-trail" onclick="viewAuditTrail('${alert.alert_id}')">Audit</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}

// 4. Care Team Panel (Section 8)
function renderCareTeamList() {
  const container = document.getElementById("care-team-list");
  if (!container) return;

  const team = appState.care_team || [];
  container.innerHTML = team
    .map((member) => {
      let dotClass = "dot-available";
      let toggleBtnText = "Set Offline";
      let nextStatus = "Offline";

      if (member.status === "Responding") {
        dotClass = "dot-responding";
        toggleBtnText = "Busy";
        nextStatus = "Available";
      } else if (member.status === "Offline") {
        dotClass = "dot-offline";
        toggleBtnText = "Set Available";
        nextStatus = "Available";
      }

      const activeBadge = member.active_alert_id
        ? `<span style="font-size: 0.7rem; color: #fbbf24; margin-left: 6px;">(Attending ${member.active_alert_id})</span>`
        : "";

      return `
        <div class="care-member-card">
          <div class="member-left">
            <span class="status-indicator-dot ${dotClass}"></span>
            <div class="member-info">
              <span class="member-name">${escapeHtml(member.name)} ${activeBadge}</span>
              <span class="member-meta">${member.role} • ${member.extension} • <strong>${member.status}</strong></span>
            </div>
          </div>
          <button class="member-status-toggle" onclick="toggleNurseStatus('${member.id}', '${nextStatus}')">
            ${toggleBtnText}
          </button>
        </div>
      `;
    })
    .join("");
}

// 5. Alert History / Audit Section (Section 14)
function renderResolvedAlertsTable() {
  const tbody = document.getElementById("resolved-alerts-tbody");
  if (!tbody) return;

  const resolved = appState.resolved_alerts || [];
  if (resolved.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" class="empty-placeholder">
          No resolved alerts logged yet for today.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = resolved
    .map((alert) => {
      const respTime = alert.response_duration_seconds != null ? `${alert.response_duration_seconds}s` : "N/A";
      const totalTime = alert.total_duration_seconds != null ? `${alert.total_duration_seconds}s` : "N/A";
      const responder = alert.responding_nurse || alert.resolving_person || "Care Team";

      return `
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 700; color: #38bdf8;">${alert.alert_id}</td>
          <td class="patient-cell">${escapeHtml(alert.patient_id)}</td>
          <td class="room-cell">${escapeHtml(alert.room_bed)}</td>
          <td style="font-family: var(--font-mono); font-size: 0.8rem;">${alert.detected_time_str}</td>
          <td><strong>${escapeHtml(responder)}</strong></td>
          <td style="font-family: var(--font-mono);">${respTime}</td>
          <td style="font-family: var(--font-mono);">${totalTime}</td>
          <td>
            <button class="btn-view-trail" onclick="viewAuditTrail('${alert.alert_id}', true)">
              View Trail
            </button>
          </td>
        </tr>
      `;
    })
    .join("");
}

// Fast Timer Sync
function updateTimersOnly(updatedAlerts) {
  for (const alert of updatedAlerts) {
    const sec = alert.timer_seconds_remaining ?? 0;
    const formatted = sec < 10 ? `0${sec}` : `${sec}`;

    // Table timer
    const tblTimer = document.getElementById(`tbl-timer-${alert.alert_id}`);
    if (tblTimer) {
      tblTimer.textContent = `00:${formatted}`;
    }

    // Prominent card timer
    const promTimer = document.getElementById("prominent-countdown");
    if (promTimer) {
      promTimer.textContent = `00:${formatted}`;
    }
  }
}

// User Actions (REST Calls)
async function respondAlert(alertId, responderName) {
  try {
    const res = await fetch(`/api/alerts/${alertId}/respond`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ responder_name: responderName }),
    });
    if (res.ok) {
      await fetchInitialState();
    }
  } catch (e) {
    console.error("Error responding to alert:", e);
  }
}

async function resolveAlert(alertId, resolverName) {
  try {
    const res = await fetch(`/api/alerts/${alertId}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        resolver_name: resolverName,
        notes: "Patient checked, vital signs stable, assisted safely.",
      }),
    });
    if (res.ok) {
      await fetchInitialState();
    }
  } catch (e) {
    console.error("Error resolving alert:", e);
  }
}

async function simulateFallAlert() {
  try {
    const res = await fetch("/api/alerts/simulate", { method: "POST" });
    if (res.ok) {
      await fetchInitialState();
    }
  } catch (e) {
    console.error("Error simulating alert:", e);
  }
}

async function resetDemo() {
  try {
    const res = await fetch("/api/alerts/reset", { method: "POST" });
    if (res.ok) {
      await fetchInitialState();
    }
  } catch (e) {
    console.error("Error resetting demo:", e);
  }
}

async function toggleNurseStatus(memberId, newStatus) {
  try {
    const res = await fetch(`/api/care-team/${memberId}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: newStatus }),
    });
    if (res.ok) {
      await fetchInitialState();
    }
  } catch (e) {
    console.error("Error toggling nurse status:", e);
  }
}

async function onTimeoutSelectChange(e) {
  const seconds = parseInt(e.target.value, 10);
  try {
    const res = await fetch("/api/timeout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ timeout_seconds: seconds }),
    });
    if (res.ok) {
      const data = await res.json();
      appState.timeout_seconds = data.timeout_seconds;
      console.log("Updated response timeout to:", data.timeout_seconds);
    }
  } catch (err) {
    console.error("Error updating timeout:", err);
  }
}

// Audit Trail Modal (Section 14)
function viewAuditTrail(alertId, isResolved = false) {
  const alertList = isResolved ? appState.resolved_alerts : appState.active_alerts;
  const alert = alertList.find((a) => a.alert_id === alertId);
  if (!alert) return;

  const modal = document.getElementById("audit-modal");
  const modalTitle = document.getElementById("modal-title");
  const modalBody = document.getElementById("modal-timeline-body");

  modalTitle.textContent = `Audit Trail — ${alert.alert_id} (${alert.patient_id}, ${alert.room_bed})`;

  const history = alert.escalation_history || [];
  if (history.length === 0) {
    modalBody.innerHTML = `<p style="color: var(--text-dim);">No history entries logged.</p>`;
  } else {
    modalBody.innerHTML = `
      <div class="timeline-list">
        ${history
          .map((item) => {
            let itemClass = "created";
            if (item.event_type === "ESCALATED") itemClass = "escalated";
            if (item.event_type === "NO_RESPONSE") itemClass = "no-response";
            if (item.event_type === "ACKNOWLEDGED") itemClass = "responded";
            if (item.event_type === "DOCTOR_ESCALATION" || item.event_type === "DOCTOR_ACKNOWLEDGED") itemClass = "doctor";
            if (item.event_type === "RESOLVED") itemClass = "resolved";

            return `
              <div class="timeline-item ${itemClass}">
                <div class="timeline-time">${item.time_str} • ${escapeHtml(item.actor)}</div>
                <div class="timeline-text">${escapeHtml(item.message)}</div>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  }

  modal.classList.add("show");
}

function closeAuditModal() {
  const modal = document.getElementById("audit-modal");
  if (modal) modal.classList.remove("show");
}

// Sound Synthesizer (Hospital Telemetry Alert)
function playTelemetryChime(type = "critical") {
  if (!soundEnabled) return;
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!audioCtx) audioCtx = new AudioContext();
    if (audioCtx.state === "suspended") audioCtx.resume();

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    const now = audioCtx.currentTime;

    if (type === "critical" || type === "escalation") {
      // Urgent double beep
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, now); // A5
      osc.frequency.setValueAtTime(659.25, now + 0.12); // E5
      gain.gain.setValueAtTime(0.18, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

      osc.start(now);
      osc.stop(now + 0.35);
    } else if (type === "attended") {
      // Pleasant acknowledgment chime
      osc.type = "sine";
      osc.frequency.setValueAtTime(523.25, now); // C5
      osc.frequency.setValueAtTime(659.25, now + 0.12); // E5
      gain.gain.setValueAtTime(0.15, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);

      osc.start(now);
      osc.stop(now + 0.3);
    }
  } catch (e) {
    // Audio autostart policy
  }
}

function toggleSound() {
  soundEnabled = !soundEnabled;
  const btn = document.getElementById("btn-sound-toggle");
  if (btn) {
    btn.innerHTML = soundEnabled ? "🔊 Sound ON" : "🔇 Sound Muted";
  }
}

// Setup Event Listeners
function setupEventListeners() {
  document.getElementById("btn-simulate-alert")?.addEventListener("click", simulateFallAlert);
  document.getElementById("btn-reset-demo")?.addEventListener("click", resetDemo);
  document.getElementById("btn-sound-toggle")?.addEventListener("click", toggleSound);
  document.getElementById("timeout-select")?.addEventListener("change", onTimeoutSelectChange);
  document.getElementById("modal-backdrop")?.addEventListener("click", (e) => {
    if (e.target.id === "audit-modal") closeAuditModal();
  });
  document.getElementById("btn-close-modal")?.addEventListener("click", closeAuditModal);
}

// Utility: HTML Escaping
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
