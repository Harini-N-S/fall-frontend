"""
alert_manager.py - Nurse Fall Alert & Escalation Workflow Engine.

Core state machine and escalation engine for hospital nurse stations:
    Fall Detected -> Primary Nurse (30s) -> Next Nurse (30s) -> Doctor Escalation.
"""

import time
import json
import urllib.request
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable


class AlertState(str, Enum):
    NEW = "NEW"
    WAITING_FOR_RESPONSE = "WAITING_FOR_RESPONSE"
    ESCALATING = "ESCALATING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    CARE_IN_PROGRESS = "CARE_IN_PROGRESS"
    RESOLVED = "RESOLVED"
    DOCTOR_ESCALATION = "DOCTOR_ESCALATION"
    DOCTOR_ACKNOWLEDGED = "DOCTOR_ACKNOWLEDGED"


class NurseStatus(str, Enum):
    AVAILABLE = "Available"
    RESPONDING = "Responding"
    OFFLINE = "Offline"


class CareTeamMember:
    def __init__(
        self,
        member_id: str,
        name: str,
        role: str = "Nurse",
        status: NurseStatus = NurseStatus.AVAILABLE,
        extension: str = "101",
    ):
        self.member_id = member_id
        self.name = name
        self.role = role
        self.status = status
        self.extension = extension
        self.active_alert_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.member_id,
            "name": self.name,
            "role": self.role,
            "status": self.status.value,
            "extension": self.extension,
            "active_alert_id": self.active_alert_id,
        }


class FallAlert:
    def __init__(
        self,
        alert_id: str,
        patient_id: str,
        room_bed: str,
        detected_time: Optional[float] = None,
        timeout_seconds: int = 30,
        fall_details: Optional[Dict[str, Any]] = None,
    ):
        self.alert_id = alert_id
        self.patient_id = patient_id
        self.room_bed = room_bed
        self.priority = "CRITICAL"
        self.state = AlertState.NEW
        self.timeout_seconds = timeout_seconds

        now = detected_time or time.time()
        self.detected_timestamp = now
        self.detected_time_str = datetime.fromtimestamp(now).strftime("%I:%M:%S %p")

        self.assigned_person_name: str = ""
        self.assigned_person_id: str = ""
        self.assigned_person_role: str = ""
        self.escalation_index: int = 0  # Index into available nurses

        self.timer_seconds_remaining: int = timeout_seconds
        self.timer_active: bool = False

        self.response_timestamp: Optional[float] = None
        self.response_time_str: Optional[str] = None
        self.responding_nurse: Optional[str] = None

        self.resolution_timestamp: Optional[float] = None
        self.resolution_time_str: Optional[str] = None
        self.resolving_person: Optional[str] = None

        self.fall_details = fall_details or {}
        self.escalation_history: List[Dict[str, Any]] = []

    def add_history(self, event_type: str, actor: str, message: str):
        now_ts = time.time()
        time_str = datetime.fromtimestamp(now_ts).strftime("%I:%M:%S %p")
        self.escalation_history.append({
            "timestamp": now_ts,
            "time_str": time_str,
            "event_type": event_type,
            "actor": actor,
            "message": message,
        })

    def to_dict(self) -> Dict[str, Any]:
        resp_duration = None
        if self.response_timestamp:
            resp_duration = round(self.response_timestamp - self.detected_timestamp, 1)

        total_duration = None
        if self.resolution_timestamp:
            total_duration = round(self.resolution_timestamp - self.detected_timestamp, 1)

        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "room_bed": self.room_bed,
            "priority": self.priority,
            "state": self.state.value,
            "detected_timestamp": self.detected_timestamp,
            "detected_time_str": self.detected_time_str,
            "assigned_person_name": self.assigned_person_name,
            "assigned_person_id": self.assigned_person_id,
            "assigned_person_role": self.assigned_person_role,
            "escalation_index": self.escalation_index,
            "timer_seconds_remaining": self.timer_seconds_remaining,
            "timer_active": self.timer_active,
            "response_timestamp": self.response_timestamp,
            "response_time_str": self.response_time_str,
            "responding_nurse": self.responding_nurse,
            "response_duration_seconds": resp_duration,
            "resolution_timestamp": self.resolution_timestamp,
            "resolution_time_str": self.resolution_time_str,
            "resolving_person": self.resolving_person,
            "total_duration_seconds": total_duration,
            "fall_details": self.fall_details,
            "escalation_history": list(self.escalation_history),
        }


class AlertManager:
    """
    Hospital Alert & Escalation Workflow Engine.
    Manages active fall alerts, 30-second response countdowns, sequential nurse escalation,
    doctor escalation, care-in-progress tracking, and resolution auditing.
    """

    def __init__(self, response_timeout_seconds: int = 30):
        self.lock = threading.RLock()
        self.response_timeout_seconds: int = response_timeout_seconds
        self.alert_counter: int = 1

        # Care Team Roster
        self.care_team: Dict[str, CareTeamMember] = {
            "nurse_priya": CareTeamMember("nurse_priya", "Nurse Priya", "Nurse", NurseStatus.AVAILABLE, "Ext. 201"),
            "nurse_anitha": CareTeamMember("nurse_anitha", "Nurse Anitha", "Nurse", NurseStatus.AVAILABLE, "Ext. 202"),
            "nurse_kavya": CareTeamMember("nurse_kavya", "Nurse Kavya", "Nurse", NurseStatus.AVAILABLE, "Ext. 203"),
            "nurse_divya": CareTeamMember("nurse_divya", "Nurse Divya", "Nurse", NurseStatus.OFFLINE, "Ext. 204"),
            "doc_suresh": CareTeamMember("doc_suresh", "Dr. Suresh (On-Call)", "Doctor", NurseStatus.AVAILABLE, "Ext. 911"),
        }

        # Sequential nurse escalation order
        self.nurse_escalation_order = ["nurse_priya", "nurse_anitha", "nurse_kavya", "nurse_divya"]
        self.doctor_id = "doc_suresh"

        # Alerts Store
        self.active_alerts: Dict[str, FallAlert] = {}
        self.resolved_alerts: List[FallAlert] = []

        # Broadcast callbacks
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []

        # Background countdown thread
        self.running: bool = True
        self.timer_thread = threading.Thread(target=self._countdown_loop, daemon=True)
        self.timer_thread.start()

    def set_timeout_seconds(self, seconds: int):
        with self.lock:
            self.response_timeout_seconds = max(2, int(seconds))

    def register_listener(self, callback: Callable[[Dict[str, Any]], None]):
        self.listeners.append(callback)

    def _broadcast(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        payload = {
            "event": event_type,
            "timestamp": time.time(),
            "time_str": datetime.now().strftime("%I:%M:%S %p"),
            "data": data or {},
            "summary": self.get_summary(),
        }
        for listener in list(self.listeners):
            try:
                listener(payload)
            except Exception:
                pass

    def get_available_nurses(self) -> List[CareTeamMember]:
        """Returns nurses in priority order who are Available."""
        available = []
        for n_id in self.nurse_escalation_order:
            nurse = self.care_team.get(n_id)
            if nurse and nurse.status == NurseStatus.AVAILABLE:
                available.append(nurse)
        return available

    def create_fall_alert(
        self,
        patient_id: str,
        room_bed: str,
        timestamp: Optional[float] = None,
        fall_details: Optional[Dict[str, Any]] = None,
    ) -> FallAlert:
        """
        Public entry point called when a confirmed fall is detected.
        Initiates the critical fall alert and alerts the primary nurse.
        """
        with self.lock:
            alert_id = f"FALL-2026-{self.alert_counter:03d}"
            self.alert_counter += 1

            alert = FallAlert(
                alert_id=alert_id,
                patient_id=patient_id,
                room_bed=room_bed,
                detected_time=timestamp,
                timeout_seconds=self.response_timeout_seconds,
                fall_details=fall_details,
            )

            # Assign primary available nurse
            avail_nurses = self.get_available_nurses()
            if avail_nurses:
                primary = avail_nurses[0]
                alert.assigned_person_id = primary.member_id
                alert.assigned_person_name = primary.name
                alert.assigned_person_role = primary.role
                alert.escalation_index = 0
                alert.state = AlertState.WAITING_FOR_RESPONSE
                alert.timer_active = True
                alert.timer_seconds_remaining = self.response_timeout_seconds

                alert.add_history(
                    "ALERT_CREATED",
                    primary.name,
                    f"Fall detected. Primary {primary.name} alerted (30s response window).",
                )
            else:
                # No nurses available -> Immediate Doctor Escalation
                doctor = self.care_team.get(self.doctor_id)
                alert.assigned_person_id = doctor.member_id if doctor else "doctor"
                alert.assigned_person_name = doctor.name if doctor else "On-Call Physician"
                alert.assigned_person_role = "Doctor"
                alert.escalation_index = len(self.nurse_escalation_order)
                alert.state = AlertState.DOCTOR_ESCALATION
                alert.timer_active = True
                alert.timer_seconds_remaining = self.response_timeout_seconds

                alert.add_history(
                    "DOCTOR_ESCALATION",
                    alert.assigned_person_name,
                    "Fall detected. All nurses unavailable. Immediately escalated to Doctor.",
                )

            self.active_alerts[alert_id] = alert

        self._broadcast("ALERT_CREATED", {"alert": alert.to_dict()})
        return alert

    def respond(self, alert_id: str, responder_name: Optional[str] = None) -> Optional[FallAlert]:
        """
        Called when a nurse or doctor clicks RESPOND.
        Stops the timer immediately and transitions to ACKNOWLEDGED / CARE_IN_PROGRESS.
        """
        with self.lock:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return None

            now = time.time()
            now_str = datetime.fromtimestamp(now).strftime("%I:%M:%S %p")

            # Immediately stop the countdown timer
            alert.timer_active = False
            alert.response_timestamp = now
            alert.response_time_str = now_str

            # Determine responder name and update care team status
            if responder_name:
                resp_name = responder_name
            elif alert.assigned_person_name:
                resp_name = alert.assigned_person_name
            else:
                resp_name = "Assigned Nurse"

            alert.responding_nurse = resp_name

            # Update care team member status to Responding
            for member in self.care_team.values():
                if member.name == resp_name or member.member_id == alert.assigned_person_id:
                    member.status = NurseStatus.RESPONDING
                    member.active_alert_id = alert_id
                    break

            if alert.state == AlertState.DOCTOR_ESCALATION:
                alert.state = AlertState.DOCTOR_ACKNOWLEDGED
                alert.add_history(
                    "DOCTOR_ACKNOWLEDGED",
                    resp_name,
                    f"Doctor acknowledged: {resp_name} responded. Escalation stopped. Care team notified.",
                )
            else:
                alert.state = AlertState.ACKNOWLEDGED
                alert.add_history(
                    "ACKNOWLEDGED",
                    resp_name,
                    f"Alert acknowledged by {resp_name}. Escalation stopped. Patient is being attended.",
                )

        self._broadcast("ALERT_RESPONDED", {"alert": alert.to_dict()})
        return alert

    def start_care(self, alert_id: str) -> Optional[FallAlert]:
        """Transitions alert state from ACKNOWLEDGED to CARE_IN_PROGRESS."""
        with self.lock:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return None
            if alert.state in (AlertState.ACKNOWLEDGED, AlertState.DOCTOR_ACKNOWLEDGED):
                alert.state = AlertState.CARE_IN_PROGRESS
                alert.add_history(
                    "CARE_IN_PROGRESS",
                    alert.responding_nurse or alert.assigned_person_name or "Nurse",
                    f"Hands-on patient care in progress by {alert.responding_nurse or 'Care Team'}.",
                )
        self._broadcast("ALERT_CARE_IN_PROGRESS", {"alert": alert.to_dict()})
        return alert

    def resolve(self, alert_id: str, resolver_name: Optional[str] = None, notes: str = "") -> Optional[FallAlert]:
        """
        Called when care is completed and the alert is marked RESOLVED.
        Moves alert from Active to Resolved History.
        """
        with self.lock:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return None

            now = time.time()
            now_str = datetime.fromtimestamp(now).strftime("%I:%M:%S %p")

            alert.state = AlertState.RESOLVED
            alert.timer_active = False
            alert.resolution_timestamp = now
            alert.resolution_time_str = now_str
            alert.resolving_person = resolver_name or alert.responding_nurse or alert.assigned_person_name or "Nurse"

            # Return nurse status back to Available
            for member in self.care_team.values():
                if member.active_alert_id == alert_id:
                    member.status = NurseStatus.AVAILABLE
                    member.active_alert_id = None

            alert.add_history(
                "RESOLVED",
                alert.resolving_person,
                f"Patient care completed by {alert.resolving_person}. Alert marked RESOLVED. {notes}".strip(),
            )

            # Move from active to resolved
            del self.active_alerts[alert_id]
            self.resolved_alerts.insert(0, alert)

        self._broadcast("ALERT_RESOLVED", {"alert": alert.to_dict()})
        return alert

    def set_nurse_status(self, member_id: str, status: NurseStatus) -> bool:
        """Configures availability of care team members (e.g. marking a nurse offline)."""
        with self.lock:
            member = self.care_team.get(member_id)
            if not member:
                return False
            member.status = status
        self._broadcast("NURSE_STATUS_CHANGED", {"care_team": [m.to_dict() for m in self.care_team.values()]})
        return True

    def reset_demo(self):
        """Resets all demo alerts and restores care team availability."""
        with self.lock:
            self.active_alerts.clear()
            self.resolved_alerts.clear()
            self.alert_counter = 1
            for member in self.care_team.values():
                member.status = NurseStatus.OFFLINE if member.member_id == "nurse_divya" else NurseStatus.AVAILABLE
                member.active_alert_id = None
        self._broadcast("DEMO_RESET", {})

    def _countdown_loop(self):
        """
        Background loop executing every 1.0 second.
        Drives the 30-second countdown for each active alert independently,
        and triggers automatic escalation upon timer expiration.
        """
        while self.running:
            time.sleep(1.0)
            escalated_alerts = []
            tick_updates = False

            with self.lock:
                for alert in list(self.active_alerts.values()):
                    if not alert.timer_active:
                        continue

                    tick_updates = True
                    alert.timer_seconds_remaining -= 1

                    # Check if response window has expired
                    if alert.timer_seconds_remaining <= 0:
                        escalated_alerts.append(self._escalate_alert(alert))

            if tick_updates:
                self._broadcast("TIMER_TICK", {
                    "alerts": [a.to_dict() for a in self.active_alerts.values()]
                })

            for esc in escalated_alerts:
                if esc:
                    self._broadcast("ALERT_ESCALATED", {"alert": esc.to_dict()})

    def _escalate_alert(self, alert: FallAlert) -> FallAlert:
        """
        Internal transition logic when the response timer expires.
        Escalates: Nurse 1 -> Nurse 2 -> Nurse 3 -> Doctor.
        """
        prev_name = alert.assigned_person_name or "Previous Nurse"
        next_index = alert.escalation_index + 1

        # Look for next available nurse
        next_nurse = None
        while next_index < len(self.nurse_escalation_order):
            cand_id = self.nurse_escalation_order[next_index]
            cand = self.care_team.get(cand_id)
            if cand and cand.status == NurseStatus.AVAILABLE:
                next_nurse = cand
                break
            next_index += 1

        # Record timeout for the previous responder
        alert.add_history(
            "NO_RESPONSE",
            prev_name,
            f"{prev_name} — No response",
        )

        if next_nurse:
            # Escalate to next nurse
            alert.state = AlertState.WAITING_FOR_RESPONSE
            alert.assigned_person_id = next_nurse.member_id
            alert.assigned_person_name = next_nurse.name
            alert.assigned_person_role = next_nurse.role
            alert.escalation_index = next_index
            alert.timer_seconds_remaining = self.response_timeout_seconds
            alert.timer_active = True

            alert.add_history(
                "ESCALATED",
                next_nurse.name,
                f"Escalated from {prev_name} → {next_nurse.name} (30s window).",
            )
        else:
            # All nurses timed out or unavailable -> ESCALATE TO DOCTOR
            doctor = self.care_team.get(self.doctor_id)
            doc_name = doctor.name if doctor else "On-Call Physician"
            doc_id = doctor.member_id if doctor else "doc_suresh"

            alert.state = AlertState.DOCTOR_ESCALATION
            alert.assigned_person_id = doc_id
            alert.assigned_person_name = doc_name
            alert.assigned_person_role = "Doctor"
            alert.escalation_index = len(self.nurse_escalation_order)
            alert.timer_seconds_remaining = self.response_timeout_seconds
            alert.timer_active = True  # Doctor alert window

            alert.add_history(
                "DOCTOR_ESCALATION",
                doc_name,
                f"NO NURSE RESPONDED. Critical escalation to Doctor: {doc_name} alerted.",
            )

        return alert

    def get_summary(self) -> Dict[str, Any]:
        """Calculates real-time top-card summary metrics."""
        with self.lock:
            active_list = list(self.active_alerts.values())
            resolved_list = list(self.resolved_alerts)

            active_critical = sum(1 for a in active_list if a.state in (
                AlertState.NEW,
                AlertState.WAITING_FOR_RESPONSE,
                AlertState.ESCALATING,
                AlertState.DOCTOR_ESCALATION,
            ))
            awaiting_response = sum(1 for a in active_list if a.state in (
                AlertState.WAITING_FOR_RESPONSE,
                AlertState.DOCTOR_ESCALATION,
            ))
            being_attended = sum(1 for a in active_list if a.state in (
                AlertState.ACKNOWLEDGED,
                AlertState.CARE_IN_PROGRESS,
                AlertState.DOCTOR_ACKNOWLEDGED,
            ))
            resolved_today = len(resolved_list)

            return {
                "active_critical": active_critical,
                "awaiting_response": awaiting_response,
                "being_attended": being_attended,
                "resolved_today": resolved_today,
                "total_active": len(active_list),
                "timeout_seconds": self.response_timeout_seconds,
            }

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Serializes complete system state for dashboard rendering."""
        with self.lock:
            return {
                "summary": self.get_summary(),
                "active_alerts": [a.to_dict() for a in self.active_alerts.values()],
                "resolved_alerts": [a.to_dict() for a in self.resolved_alerts],
                "care_team": [m.to_dict() for m in self.care_team.values()],
                "server_time": time.time(),
                "server_time_str": datetime.now().strftime("%I:%M:%S %p"),
                "timeout_seconds": self.response_timeout_seconds,
            }


# Singleton Global Manager Instance
_global_alert_manager: Optional[AlertManager] = None
_manager_lock = threading.RLock()


def get_alert_manager(timeout_seconds: int = 30) -> AlertManager:
    global _global_alert_manager
    with _manager_lock:
        if _global_alert_manager is None:
            _global_alert_manager = AlertManager(response_timeout_seconds=timeout_seconds)
        return _global_alert_manager


def createFallAlert(
    patientId: str,
    roomId: str,
    timestamp: Optional[float] = None,
    fall_details: Optional[Dict[str, Any]] = None,
    server_url: Optional[str] = "http://127.0.0.1:8000",
) -> Dict[str, Any]:
    """
    Standard interface function as requested in Requirement 12 & Step 2:
    createFallAlert(patientId, roomId, timestamp)
    
    If the central FastAPI dashboard server is running at server_url, forwards the
    alert so connected browser WebSockets receive it in real time.
    Also creates/updates the local AlertManager instance for in-process access.
    """
    manager = get_alert_manager()
    is_server_hosting = getattr(manager, "_is_server_hosting", False)

    # When called from an external runner (e.g. run_hybrid_pipeline.py), notify the active dashboard server
    if not is_server_hosting and server_url:
        try:
            ts_val = float(timestamp) if timestamp is not None else time.time()
            clean_details = {}
            if fall_details:
                for k, v in fall_details.items():
                    if hasattr(v, "item"):
                        clean_details[k] = v.item()
                    elif isinstance(v, (int, float, str, bool)):
                        clean_details[k] = v
                    else:
                        clean_details[k] = str(v)
            payload = json.dumps({
                "patient_id": str(patientId),
                "room_bed": str(roomId),
                "timestamp": ts_val,
                "fall_details": clean_details,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{server_url}/api/alerts",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "alert" in data:
                    return data["alert"]
        except Exception:
            # Fall back to local in-process manager if dashboard server is not reachable
            pass

    alert = manager.create_fall_alert(
        patient_id=patientId,
        room_bed=roomId,
        timestamp=timestamp,
        fall_details=fall_details,
    )
    return alert.to_dict()
