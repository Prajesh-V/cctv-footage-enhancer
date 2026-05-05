from __future__ import annotations

import os
import uuid
import time
import asyncio
import threading
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import io
from fastapi import WebSocket, WebSocketDisconnect

import numpy as np

from pipeline.execution import submit, shutdown
from pipeline.cancellation import CancellationToken
from pipeline.frame import Frame
from pipeline.output_writer import DiskOutputWriter
from pipeline.interfaces import PipelineResult
from pipeline import process_image_pipeline
from video import process_video_pipeline
from pipeline.output_writer import VideoOutputWriter, DiskOutputWriter
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Monkey-patch PIL.Image.ANTIALIAS for compatibility with older libraries (realesrgan, gfpgan)
if not hasattr(Image, 'ANTIALIAS'):
    # In Pillow 10.0.0+, ANTIALIAS was removed in favor of Resampling.LANCZOS
    if hasattr(Image, 'Resampling'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    else:
        # Fallback for even older versions if needed, though unlikely here
        Image.ANTIALIAS = 1 

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
# ROOT_DIR points to the 'clarityai' project root (parent of 'backend')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT_DIR, "inputs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# Ensure base directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _cleanup_old_files():
    retention_hours = float(os.getenv("FILE_RETENTION_HOURS", "24"))
    cutoff = time.time() - (retention_hours * 3600)
    for root in [INPUT_DIR, OUTPUT_DIR]:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    if os.path.getmtime(fp) < cutoff:
                        os.remove(fp)
                except Exception:
                    pass


app = FastAPI(title="ClarityAI MVP Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cleanup_thread = None


@app.on_event("startup")
def _start_cleanup():
    global _cleanup_thread
    def _loop():
        while True:
            _cleanup_old_files()
            time.sleep(3600)
    _cleanup_thread = threading.Thread(target=_loop, daemon=True)
    _cleanup_thread.start()


JOB_STORE: Dict[str, Dict] = {}


def _new_job_id() -> str:
    return str(uuid.uuid4())


@app.post("/api/jobs/image")
async def create_image_job(image: UploadFile = File(...), upscale_factor: int = 2, enable_face_restore: bool = True, plate_detection: bool = True, roi: Optional[dict] = None):
    job_id = _new_job_id()
    input_path = os.path.join(INPUT_DIR, f"{job_id}_{image.filename}")
    with open(input_path, "wb") as f:
        content = await image.read()
        f.write(content)

    cancellation_token = CancellationToken()
    writer = DiskOutputWriter(job_id)

    JOB_STORE[job_id] = {
        "id": job_id,
        "type": "image",
        "status": "in_progress",
        "input_path": input_path,
        "output_path": None,
        "roi": roi,
        "options": {"upscale_factor": upscale_factor, "face_restore": enable_face_restore, "plate_detection": plate_detection},
        "progress": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
        "cancellation_token": cancellation_token,
    }

    def _runner():
        result: PipelineResult = process_image_pipeline(
            input_path,
            roi,
            upscale_factor,
            enable_face_restore,
            plate_detection,
            cancellation_token,
            writer,
        )
        JOB_STORE[job_id]["status"] = "completed" if not cancellation_token.cancelled else "cancelled"
        JOB_STORE[job_id]["output_path"] = result.output if result else None
        JOB_STORE[job_id]["metrics"] = result.metadata if result else {}
        JOB_STORE[job_id]["updated_at"] = time.time()
        return result

    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(_runner)

    return {"job_id": job_id, "status": JOB_STORE[job_id]["status"]}


@app.post("/api/utils/preview-frame")
async def get_video_preview(video: UploadFile = File(...)):
    """Extracts the first frame of a video and returns it as a JPEG."""
    # Temporary save to read with OpenCV
    temp_id = _new_job_id()
    temp_path = os.path.join(INPUT_DIR, f"temp_preview_{temp_id}_{video.filename}")
    
    try:
        with open(temp_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        cap = cv2.VideoCapture(temp_path)
        success, frame = cap.read()
        cap.release()
        
        if not success:
            return JSONResponse(status_code=400, content={"error": "could_not_extract_frame"})
        
        # Convert BGR to RGB and then to JPEG
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Cleanup temp file
        os.remove(temp_path)
        
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/jobs/video")
async def create_video_job(
    video: UploadFile = File(...),
    upscale_factor: int = 2,
    enable_face_restore: bool = True,
    plate_detection: bool = True,
    roi: Optional[str] = Form(None),
):
    video.file.seek(0, 2)
    file_size = video.file.tell()
    video.file.seek(0)
    if file_size > 200 * 1024 * 1024:
        return {"error": "file_too_large", "max_size_mb": 200}

    parsed_roi = None
    if roi:
        try:
            parsed_roi = json.loads(roi)
        except Exception:
            parsed_roi = None
    roi = parsed_roi

    job_id = _new_job_id()
    input_path = os.path.join(INPUT_DIR, f"{job_id}_{video.filename}")
    with open(input_path, "wb") as f:
        content = await video.read()
        f.write(content)

    cancellation_token = CancellationToken()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": "cannot_open_input_video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    def _progress_cb(current, total):
        if job_id in JOB_STORE:
            JOB_STORE[job_id]['frames_processed'] = int(current)
            JOB_STORE[job_id]['frames_total'] = int(total) if total else 0
            if total:
                JOB_STORE[job_id]['progress'] = int((current / float(total)) * 100)
            JOB_STORE[job_id]['updated_at'] = time.time()

    # Pass 0 for width and height to allow the writer to discover the size from the first processed frame
    writer = VideoOutputWriter(job_id, fps, 0, 0, progress_callback=_progress_cb)

    JOB_STORE[job_id] = {
        "id": job_id,
        "type": "video",
        "status": "in_progress",
        "input_path": input_path,
        "output_path": None,
        "roi": roi,
        "options": {"upscale_factor": upscale_factor, "face_restore": enable_face_restore, "plate_detection": plate_detection},
        "progress": 0,
        "frames_total": 0,
        "frames_processed": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
        "cancellation_token": cancellation_token,
    }

    def _runner_video():
        try:
            result = process_video_pipeline(input_path, roi, upscale_factor, cancellation_token, writer)
            
            # Check if we actually got an output file
            output_path = result.get("output") if isinstance(result, dict) else None
            
            if output_path and os.path.exists(output_path):
                JOB_STORE[job_id]["status"] = "completed"
                JOB_STORE[job_id]["output_path"] = output_path
            else:
                JOB_STORE[job_id]["status"] = "failed"
                # Capture the errors from the pipeline if available
                pipe_errors = result.get("metadata", {}).get("errors", []) if isinstance(result, dict) else ["Unknown pipeline error"]
                JOB_STORE[job_id]["error_msg"] = pipe_errors[0] if pipe_errors else "No output generated"

            JOB_STORE[job_id]["metrics"] = result.get("metadata", {}) if isinstance(result, dict) else {}
            JOB_STORE[job_id]["updated_at"] = time.time()
            return result
        except Exception as e:
            JOB_STORE[job_id]["status"] = "failed"
            JOB_STORE[job_id]["output_path"] = None
            JOB_STORE[job_id]["updated_at"] = time.time()
            return {"error": str(e)}

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(_runner_video)

    return {"job_id": job_id, "status": JOB_STORE[job_id]["status"]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        return {"error": "job_not_found"}
    metrics = job.get("metrics", {})
    return {
        "id": job["id"],
        "type": job["type"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "created_at": job["created_at"],
        "updated_at": job.get("updated_at"),
        "roi": job.get("roi"),
        "options": job.get("options"),
        "ocr_texts": metrics.get("ocr_texts", []),
        "plates": metrics.get("plates", []),
    }


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        return {"error": "job_not_found"}
    path = job.get("output_path")
    if not path or not os.path.exists(path):
        return {"error": "result_not_ready"}
    return FileResponse(path)


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        return {"error": "job_not_found"}
    token = job.get("cancellation_token")
    if token:
        token.cancel()
    job["status"] = "cancelled"
    job["updated_at"] = time.time()
    return {"job_id": job_id, "status": job["status"]}


@app.get("/api/jobs/history")
async def list_jobs():
    return [{"id": j["id"], "type": j["type"], "status": j["status"], "created_at": j["created_at"]} for j in JOB_STORE.values()]


@app.websocket("/ws/jobs/{job_id}")
async def ws_job_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        last_progress = -1
        while True:
            job = JOB_STORE.get(job_id)
            if not job:
                await websocket.send_json({"job_id": job_id, "status": "unknown"})
                break
            status = job.get("status", "pending")
            progress = int(job.get("progress", 0))
            if progress != last_progress:
                await websocket.send_json({"job_id": job_id, "progress": progress, "status": status})
                last_progress = progress
            if status in ("completed", "cancelled", "failed"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
