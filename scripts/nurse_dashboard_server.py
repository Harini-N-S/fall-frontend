"""
Nurse Fall Alert & Escalation Dashboard Server
FastAPI backend with real-time WebSockets, REST APIs, and static hospital dashboard frontend.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add scripts directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DASHBOARD_DIR = BASE_DIR / "dashboard"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from alert_manager import (
    AlertManager,
    AlertState,
    NurseStatus,
    get_alert_manager,
    createFallAlert,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NurseDashboard")

app = FastAPI(title="Hospital Nurse Fall Alert & Escalation System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
connected_websockets: Set[WebSocket] = set()
main_loop: Optional[asyncio.AbstractEventLoop] = None
manager: AlertManager = get_alert_manager(timeout_seconds=30)
manager._is_server_hosting = True

# Request Models
class CreateAlertRequest(BaseModel):
    patient_id: str = "Patient 104"
    room_bed: str = "Ward A - Bed 12"
    timestamp: Optional[float] = None
    fall_details: Optional[Dict[str, Any]] = None

class RespondRequest(BaseModel):
    responder_name: Optional[str] = None

class ResolveRequest(BaseModel):
    resolver_name: Optional[str] = None
    notes: Optional[str] = ""

class StatusUpdateRequest(BaseModel):
    status: str

class TimeoutUpdateRequest(BaseModel):
    timeout_seconds: int


async def broadcast_ws(payload: Dict[str, Any]):
    """Broadcasts event payload to all connected WebSocket clients."""
    if not connected_websockets:
        return
    text = json.dumps(payload)
    disconnected = set()
    for ws in list(connected_websockets):
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        connected_websockets.discard(ws)


def on_alert_manager_event(payload: Dict[str, Any]):
    """Thread-safe callback from AlertManager background timer or state updates."""
    global main_loop
    if main_loop and main_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(broadcast_ws(payload), main_loop)
        except Exception as e:
            logger.debug(f"Broadcast threadsafe error: {e}")


# Register AlertManager listener
manager.register_listener(on_alert_manager_event)


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    logger.info("Nurse Alert Dashboard Server started.")


# REST Endpoints
@app.get("/api/state")
def get_state():
    """Returns current complete dashboard state."""
    return manager.get_dashboard_state()


@app.get("/api/summary")
def get_summary():
    """Returns top-level card summary counts."""
    return manager.get_summary()


@app.get("/api/history")
def get_history():
    """Returns resolved alerts audit history."""
    state = manager.get_dashboard_state()
    return {"resolved_alerts": state["resolved_alerts"]}


@app.post("/api/alerts")
def create_alert(req: CreateAlertRequest):
    """Creates a new fall alert (e.g. from fall-detection pipeline)."""
    alert = manager.create_fall_alert(
        patient_id=req.patient_id,
        room_bed=req.room_bed,
        timestamp=req.timestamp,
        fall_details=req.fall_details,
    )
    return {"status": "success", "alert": alert.to_dict()}


@app.post("/api/alerts/{alert_id}/respond")
def respond_alert(alert_id: str, req: Optional[RespondRequest] = None):
    """Nurse or Doctor clicks RESPOND."""
    responder_name = req.responder_name if req else None
    alert = manager.respond(alert_id, responder_name=responder_name)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found or already closed")
    return {"status": "success", "alert": alert.to_dict()}


@app.post("/api/alerts/{alert_id}/start-care")
def start_care(alert_id: str):
    """Transitions alert from ACKNOWLEDGED to CARE_IN_PROGRESS."""
    alert = manager.start_care(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found or invalid state")
    return {"status": "success", "alert": alert.to_dict()}


@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str, req: Optional[ResolveRequest] = None):
    """Marks alert as RESOLVED / CARE COMPLETED."""
    resolver_name = req.resolver_name if req else None
    notes = req.notes if req else ""
    alert = manager.resolve(alert_id, resolver_name=resolver_name, notes=notes)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found or already resolved")
    return {"status": "success", "alert": alert.to_dict()}


@app.post("/api/alerts/simulate")
def simulate_fall():
    """Demo button: triggers a test fall alert for demonstration."""
    sample_patients = [
        ("Patient 104", "Ward A - Bed 12"),
        ("Patient 102", "Ward B - Bed 04"),
        ("Patient 108", "Ward A - Bed 01"),
        ("Patient 115", "ICU Stepdown - Bed 07"),
    ]
    idx = (manager.alert_counter - 1) % len(sample_patients)
    p_id, r_id = sample_patients[idx]

    alert = manager.create_fall_alert(
        patient_id=p_id,
        room_bed=r_id,
        fall_details={
            "source": "SIMULATION_DEMO",
            "confidence": 0.98,
            "sensor": "Ceiling Camera #4",
        },
    )
    return {"status": "success", "alert": alert.to_dict()}


@app.post("/api/alerts/reset")
def reset_alerts():
    """Reset Demo button: clears all demo alerts and resets nurses."""
    manager.reset_demo()
    return {"status": "success", "message": "Demo state reset successfully."}


@app.post("/api/timeout")
def set_timeout(req: TimeoutUpdateRequest):
    """Updates response countdown window (e.g. 30s or fast 5s demo)."""
    manager.set_timeout_seconds(req.timeout_seconds)
    return {"status": "success", "timeout_seconds": manager.response_timeout_seconds}


@app.post("/api/care-team/{member_id}/status")
def update_nurse_status(member_id: str, req: StatusUpdateRequest):
    """Configures nurse availability (Available vs Offline)."""
    status_str = req.status.capitalize()
    if status_str not in ("Available", "Offline", "Responding"):
        raise HTTPException(status_code=400, detail="Invalid status value")
    status_enum = NurseStatus(status_str)
    success = manager.set_nurse_status(member_id, status_enum)
    if not success:
        raise HTTPException(status_code=404, detail="Care team member not found")
    return {"status": "success", "member_id": member_id, "new_status": status_str}


# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    # Send immediate state snapshot upon connection
    state = manager.get_dashboard_state()
    await websocket.send_text(json.dumps({
        "event": "INITIAL_STATE",
        "data": state,
        "summary": state["summary"],
    }))
    try:
        while True:
            # Keep connection open, handle potential ping/pong or client action
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                cmd = msg.get("action")
                if cmd == "respond":
                    manager.respond(msg.get("alert_id"), responder_name=msg.get("responder_name"))
                elif cmd == "resolve":
                    manager.resolve(msg.get("alert_id"), resolver_name=msg.get("resolver_name"), notes=msg.get("notes", ""))
                elif cmd == "simulate":
                    simulate_fall()
                elif cmd == "reset":
                    manager.reset_demo()
            except Exception as e:
                logger.error(f"WS command error: {e}")
    except WebSocketDisconnect:
        connected_websockets.discard(websocket)
    except Exception:
        connected_websockets.discard(websocket)


# Mount static assets if dashboard folder exists
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

    @app.get("/")
    def serve_index():
        index_file = DASHBOARD_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>Dashboard HTML loading...</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nurse_dashboard_server:app", host="127.0.0.1", port=8000, reload=False)
