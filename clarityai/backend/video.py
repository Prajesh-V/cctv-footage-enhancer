from __future__ import annotations

import os
import time
from typing import Optional, Dict, List, Any

import cv2
import numpy as np

from pipeline.frame import Frame
from pipeline.cancellation import CancellationToken
from pipeline.output_writer import VideoOutputWriter
from pipeline.interfaces import PipelineResult
from pipeline import _normalize_roi
from ml_runtime import MLRuntime

# ROOT_DIR points to the 'clarityai' project root (parent of 'backend')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_VIDEOS_DIR = os.path.join(ROOT_DIR, "outputs", "videos")
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)


def _process_frame(frame: Frame, roi: Optional[Dict[str, float]], upscale_factor: int, ml_runtime: MLRuntime) -> tuple[np.ndarray, Dict[str, Any]]:
    img = frame.data
    norm_roi = _normalize_roi(roi)
    enable_face_restore = frame.metadata.get("enable_face_restore", True) if hasattr(frame, 'metadata') else True
    detect_plates = frame.metadata.get("detect_plates", True) if hasattr(frame, 'metadata') else True
    # Update upscale factor on the upscaler if needed
    if upscale_factor != ml_runtime.upscaler.upscale_factor:
        ml_runtime.upscaler.upscale_factor = upscale_factor
    processed, result = ml_runtime.process_frame(img, norm_roi, enable_face_restore, detect_plates)
    return processed, result


def mux_audio(original_video: str, video_only: str, output_video: str) -> None:
    """Mux audio from original_video into video_only to produce output_video."""
    import shutil
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not installed or not in PATH.")
    audio_path = os.path.splitext(original_video)[0] + "_audio.aac"
    import subprocess
    subprocess.run(["ffmpeg", "-i", original_video, "-vn", "-acodec", "copy", audio_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-i", video_only, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", output_video], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_video_pipeline(input_path: str, roi: Optional[Dict[str, float]], upscale_factor: int, cancellation_token: CancellationToken, writer: VideoOutputWriter) -> PipelineResult:
    t0 = time.time()
    errors: List[str] = []
    frames_total = 0
    frames_processed = 0
    all_ocr_texts: List[Dict] = []
    all_plates: List[Dict] = []

    use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
    ml_runtime = MLRuntime(use_gpu=use_gpu)

    cap = cv2.VideoCapture(input_path)
    print(f"DEBUG: Attempting to open video: {input_path}")
    if not cap.isOpened():
        print(f"DEBUG: Failed to open video file: {input_path}")
        return PipelineResult(output=None, metadata={"processing_time": time.time() - t0, "roi_applied": bool(roi), "errors": ["cannot_open_input"]})
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"DEBUG: Video opened. FPS: {fps}, Size: {w}x{h}, Total Frames: {frames_total}")
    if hasattr(writer, "_total_frames"):
        try:
            writer._total_frames = frames_total
        except Exception:
            pass
    if writer is None:
        writer = VideoOutputWriter(job_id="default_job", fps=int(fps), width=0, height=0, output_root=OUTPUT_VIDEOS_DIR)
    frame_idx = 0
    success, frame = cap.read()
    while success:
        if cancellation_token.cancelled:
            break
        norm_roi = None
        if roi is not None:
            norm_roi = _normalize_roi(roi)
        try:
            print(f"DEBUG: Processing frame {frame_idx}/{frames_total}...")
            proc, result = _process_frame(Frame(data=frame, index=frame_idx, timestamp=time.time(), roi=norm_roi, metadata={"enable_face_restore": True, "detect_plates": True}), norm_roi, upscale_factor, ml_runtime)
            writer.write_frame(Frame(data=proc, index=frame_idx, timestamp=time.time(), roi=norm_roi))
            frames_processed += 1
            if result.get("ocr_texts"):
                all_ocr_texts.extend(result["ocr_texts"])
            if result.get("plates"):
                all_plates.extend(result["plates"])
        except Exception as e:
            print(f"DEBUG: Error on frame {frame_idx}: {str(e)}")
            errors.append(str(e))
        frame_idx += 1
        success, frame = cap.read()

    cap.release()
    writer.finalize()
    output_path = writer.path
    final_output = os.path.splitext(output_path)[0] + "_final.mp4"
    try:
        mux_audio(input_path, output_path, final_output)
    except Exception as e:
        errors.append(str(e))
        final_output = output_path

    metadata = {
        "processing_time": time.time() - t0,
        "roi_applied": bool(roi),
        "errors": errors,
        "frames_total": frames_total,
        "frames_processed": frames_processed,
        "ocr_texts": all_ocr_texts,
        "plates": all_plates,
    }
    return PipelineResult(output=final_output, metadata=metadata)
