"""FastAPI server for the auto-rig UI."""

import dotenv
dotenv.load_dotenv()

import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import secrets
import shutil
import sys
import tempfile
import uuid

from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Basic auth (disabled when AUTH_USER and AUTH_PASS are not set)
AUTH_USER = os.environ.get("AUTH_USER", "")
AUTH_PASS = os.environ.get("AUTH_PASS", "")
AUTH_ENABLED = bool(AUTH_USER and AUTH_PASS)

if AUTH_ENABLED:
    security = HTTPBasic()

    def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
        correct_user = hmac.compare_digest(credentials.username, AUTH_USER)
        correct_pass = hmac.compare_digest(credentials.password, AUTH_PASS)
        if not (correct_user and correct_pass):
            raise HTTPException(status_code=401, detail="Unauthorized",
                                headers={"WWW-Authenticate": "Basic"})
        return credentials.username

# Add parent dir to path so we can import processing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google import genai

from processing import diecut, bboxes, rig
from processing.simple_background import remove_background_simple

app_deps = [Depends(check_auth)] if AUTH_ENABLED else []
app = FastAPI(title="Auto-Rig UI", dependencies=app_deps)

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Session storage
sessions = {}

PART_NAMES = [
    "head", "torso",
    "left_upperarm", "right_upperarm",
    "left_forearm", "right_forearm",
    "left_hand", "right_hand",
    "left_thigh", "right_thigh",
    "left_calf", "right_calf",
    "left_foot", "right_foot",
]


def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


def create_session():
    session_id = str(uuid.uuid4())[:8]
    work_dir = tempfile.mkdtemp(prefix=f"autorig_{session_id}_")
    sessions[session_id] = {
        "id": session_id,
        "work_dir": work_dir,
        "image_path": None,
        "diecut_path": None,
        "parts_dir": None,
        "rig_path": None,
        "status": "created",
    }
    return sessions[session_id]


@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)


@app.post("/api/session")
async def new_session():
    session = create_session()
    return {"session_id": session["id"]}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...), session_id: str = ""):
    if not session_id:
        session = create_session()
        session_id = session["id"]
    else:
        session = get_session(session_id)

    image_path = os.path.join(session["work_dir"], "original.png")
    with open(image_path, "wb") as f:
        content = await file.read()
        f.write(content)

    session["image_path"] = image_path
    session["status"] = "uploaded"

    # Return the uploaded image as base64 for preview
    b64 = base64.b64encode(content).decode("ascii")
    return {
        "session_id": session_id,
        "status": "uploaded",
        "image_preview": f"data:image/png;base64,{b64}",
    }


@app.post("/api/diecut")
async def run_diecut(session_id: str, rounds: int = 5, diecut_model: Optional[str] = None, vision_model: Optional[str] = None):
    session = get_session(session_id)
    if not session["image_path"]:
        raise HTTPException(status_code=400, detail="No image uploaded")

    session["status"] = "running_diecut"
    diecut_path = os.path.join(session["work_dir"], "diecut.png")

    kwargs = {}
    if diecut_model:
        kwargs["model"] = diecut_model
    if vision_model:
        kwargs["judge_model"] = vision_model

    try:
        client = genai.Client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: diecut(client, session["image_path"], diecut_path, False, rounds, **kwargs),
        )
        session["diecut_path"] = diecut_path
        session["status"] = "diecut_done"

        with open(diecut_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return {
            "status": "diecut_done",
            "diecut_preview": f"data:image/png;base64,{b64}",
        }
    except Exception as e:
        session["status"] = "diecut_error"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bboxes")
async def run_bboxes(session_id: str, vision_model: Optional[str] = None):
    session = get_session(session_id)
    if not session["diecut_path"]:
        raise HTTPException(status_code=400, detail="Diecut not done yet")

    session["status"] = "running_bboxes"
    parts_dir = os.path.join(session["work_dir"], "parts")
    os.makedirs(parts_dir, exist_ok=True)

    kwargs = {}
    if vision_model:
        kwargs["model"] = vision_model

    try:
        client = genai.Client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: bboxes(client, session["diecut_path"], parts_dir, **kwargs),
        )
        session["parts_dir"] = parts_dir
        session["status"] = "bboxes_done"
        return {"status": "bboxes_done", "parts": _list_parts(parts_dir)}
    except Exception as e:
        session["status"] = "bboxes_error"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/remove-bg")
async def run_remove_bg(session_id: str):
    session = get_session(session_id)
    if not session["parts_dir"]:
        raise HTTPException(status_code=400, detail="Bboxes not done yet")

    session["status"] = "running_bg_removal"
    parts_dir = session["parts_dir"]

    try:
        loop = asyncio.get_event_loop()
        for name in PART_NAMES:
            path = os.path.join(parts_dir, f"{name}.png")
            if os.path.exists(path):
                await loop.run_in_executor(None, remove_background_simple, path, 30)

        session["status"] = "bg_removal_done"
        return {"status": "bg_removal_done", "parts": _list_parts(parts_dir)}
    except Exception as e:
        session["status"] = "bg_removal_error"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rig")
async def run_rig(session_id: str, rig_model: Optional[str] = None):
    session = get_session(session_id)
    if not session["parts_dir"]:
        raise HTTPException(status_code=400, detail="Parts not ready")

    session["status"] = "running_rig"
    rig_path = os.path.join(session["work_dir"], "rig.json")

    kwargs = {}
    if rig_model:
        kwargs["model"] = rig_model

    try:
        client = genai.Client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: rig(client, session["image_path"], session["parts_dir"], rig_path, **kwargs),
        )
        session["rig_path"] = rig_path
        session["status"] = "rig_done"

        with open(rig_path, "r") as f:
            rig_data = json.load(f)
        return {"status": "rig_done", "rig": rig_data}
    except Exception as e:
        session["status"] = "rig_error"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-all")
async def run_all(session_id: str, rounds: int = 5, diecut_model: Optional[str] = None, vision_model: Optional[str] = None, rig_model: Optional[str] = None):
    """Run the full pipeline: diecut → bboxes → bg removal → rig."""
    session = get_session(session_id)
    if not session["image_path"]:
        raise HTTPException(status_code=400, detail="No image uploaded")

    results = {}
    print(f"[pipeline] Starting full pipeline (rounds={rounds}) for session {session_id}")

    diecut_kwargs = {}
    if diecut_model:
        diecut_kwargs["model"] = diecut_model
    if vision_model:
        diecut_kwargs["judge_model"] = vision_model

    bbox_kwargs = {}
    if vision_model:
        bbox_kwargs["model"] = vision_model

    rig_kwargs = {}
    if rig_model:
        rig_kwargs["model"] = rig_model

    # Step 1: Diecut
    print("[pipeline] Step 1/4: Diecut")
    session["status"] = "running_diecut"
    diecut_path = os.path.join(session["work_dir"], "diecut.png")
    try:
        client = genai.Client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: diecut(client, session["image_path"], diecut_path, False, rounds, **diecut_kwargs),
        )
        session["diecut_path"] = diecut_path
        results["diecut"] = "done"
        print("[pipeline] Step 1/4: Diecut done")
    except Exception as e:
        session["status"] = "diecut_error"
        raise HTTPException(status_code=500, detail=f"Diecut failed: {e}")

    # Step 2: Bboxes
    print("[pipeline] Step 2/4: Bboxes")
    session["status"] = "running_bboxes"
    parts_dir = os.path.join(session["work_dir"], "parts")
    os.makedirs(parts_dir, exist_ok=True)
    try:
        await loop.run_in_executor(
            None, lambda: bboxes(client, session["diecut_path"], parts_dir, **bbox_kwargs),
        )
        session["parts_dir"] = parts_dir
        results["bboxes"] = "done"
        print("[pipeline] Step 2/4: Bboxes done")
    except Exception as e:
        session["status"] = "bboxes_error"
        raise HTTPException(status_code=500, detail=f"Bboxes failed: {e}")

    # Step 3: Background removal
    print("[pipeline] Step 3/4: Background removal")
    session["status"] = "running_bg_removal"
    try:
        for name in PART_NAMES:
            path = os.path.join(parts_dir, f"{name}.png")
            if os.path.exists(path):
                await loop.run_in_executor(
                    None, remove_background_simple, path, 30
                )
        results["bg_removal"] = "done"
        print("[pipeline] Step 3/4: Background removal done")
    except Exception as e:
        session["status"] = "bg_removal_error"
        raise HTTPException(
            status_code=500, detail=f"Background removal failed: {e}"
        )

    # Step 4: Rig generation
    print("[pipeline] Step 4/4: Rig generation")
    session["status"] = "running_rig"
    rig_path = os.path.join(session["work_dir"], "rig.json")
    try:
        await loop.run_in_executor(
            None,
            lambda: rig(client, session["image_path"], session["parts_dir"], rig_path, **rig_kwargs),
        )
        session["rig_path"] = rig_path
        results["rig"] = "done"
        print("[pipeline] Step 4/4: Rig generation done")
    except Exception as e:
        session["status"] = "rig_error"
        raise HTTPException(status_code=500, detail=f"Rig generation failed: {e}")

    session["status"] = "complete"
    print("[pipeline] Pipeline complete!")

    with open(rig_path, "r") as f:
        rig_data = json.load(f)

    return {
        "status": "complete",
        "results": results,
        "parts": _list_parts(parts_dir),
        "rig": rig_data,
    }


@app.get("/api/parts")
async def list_parts(session_id: str):
    session = get_session(session_id)
    if not session["parts_dir"]:
        return {"parts": []}
    return {"parts": _list_parts(session["parts_dir"])}


@app.get("/api/parts/{name}")
async def get_part(name: str, session_id: str):
    session = get_session(session_id)
    if not session["parts_dir"]:
        raise HTTPException(status_code=404, detail="No parts available")
    path = os.path.join(session["parts_dir"], f"{name}.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Part {name} not found")
    return FileResponse(path, media_type="image/png")


@app.put("/api/parts/{name}")
async def update_part(name: str, session_id: str, file: UploadFile = File(...)):
    session = get_session(session_id)
    if not session["parts_dir"]:
        raise HTTPException(status_code=400, detail="No parts directory")

    path = os.path.join(session["parts_dir"], f"{name}.png")
    with open(path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Regenerate rig with updated part
    return {"status": "updated", "part": name}


@app.get("/api/rig-data")
async def get_rig_data(session_id: str):
    session = get_session(session_id)
    if not session["rig_path"] or not os.path.exists(session["rig_path"]):
        raise HTTPException(status_code=404, detail="No rig available")
    with open(session["rig_path"], "r") as f:
        return json.load(f)


class RigUpdate(BaseModel):
    rig: dict


@app.put("/api/rig-data")
async def update_rig_data(session_id: str, update: RigUpdate):
    session = get_session(session_id)
    if not session["rig_path"]:
        # Create new rig path
        session["rig_path"] = os.path.join(session["work_dir"], "rig.json")

    with open(session["rig_path"], "w") as f:
        json.dump(update.rig, f, indent=2)

    return {"status": "updated"}


@app.post("/api/regenerate-rig")
async def regenerate_rig(session_id: str, rig_model: Optional[str] = None):
    """Regenerate rig from current parts without re-running diecut/bboxes."""
    session = get_session(session_id)
    if not session["parts_dir"] or not session["image_path"]:
        raise HTTPException(status_code=400, detail="Parts or image not available")

    kwargs = {}
    if rig_model:
        kwargs["model"] = rig_model

    rig_path = os.path.join(session["work_dir"], "rig.json")
    try:
        client = genai.Client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: rig(client, session["image_path"], session["parts_dir"], rig_path, **kwargs),
        )
        session["rig_path"] = rig_path
        with open(rig_path, "r") as f:
            rig_data = json.load(f)
        return {"status": "rig_done", "rig": rig_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _list_parts(parts_dir):
    """List parts with base64 thumbnails."""
    parts = []
    for name in PART_NAMES:
        path = os.path.join(parts_dir, f"{name}.png")
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            parts.append({
                "name": name,
                "image": f"data:image/png;base64,{b64}",
            })
    return parts


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
