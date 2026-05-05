from __future__ import annotations

import os
import time
from typing import Optional, Dict
import numpy as np
from PIL import Image, ImageDraw

from .frame import Frame
from .cancellation import CancellationToken
from .output_writer import DiskOutputWriter
from .interfaces import PipelineResult


def process_image_pipeline(
    input_path: str,
    roi: Optional[Dict[str, float]],
    upscale_factor: int,
    enable_face_restore: bool,
    detect_plates: bool,
    cancellation_token: CancellationToken,
    writer: DiskOutputWriter,
) -> PipelineResult:
    t0 = time.time()
    errors = []

    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        errors.append(str(e))
        return PipelineResult(output=None, metadata={"processing_time": time.time() - t0, "roi_applied": False, "errors": errors})

    w, h = img.width, img.height
    new_size = (int(w * upscale_factor), int(h * upscale_factor))
    upscaled = img.resize(new_size, Image.LANCZOS)

    # Check cancellation before heavy work
    if cancellation_token.cancelled:
        return PipelineResult(output=None, metadata={"processing_time": time.time() - t0, "roi_applied": bool(roi), "errors": ["cancelled"]})

    # ROI processing (simulate by drawing ROI boundary on the upscaled image)
    roi_applied = False
    rect = None
    if roi:
        roi_applied = True
        rx = int(roi.get("x", 0.0) * new_size[0])
        ry = int(roi.get("y", 0.0) * new_size[1])
        rw = int(roi.get("w", 1.0) * new_size[0])
        rh = int(roi.get("h", 1.0) * new_size[1])
        rect = [rx, ry, rx + rw, ry + rh]
        draw = ImageDraw.Draw(upscaled)
        draw.rectangle(rect, outline=(255, 0, 0), width=max(1, int(max(new_size) * 0.003)))

    # Simulated detection results (OCR and plates) when ROI is present
    ocr_texts = []
    plate_texts = []
    if roi_applied:
        if detect_plates:
            plate_texts.append({"bbox": rect, "text": "ABC1234", "conf": 0.85})
        ocr_texts.append({"region": rect, "text": "DEMO_FACE", "conf": 0.92})

    # Persist output via writer
    frame = Frame(data=np.array(upscaled), index=0, timestamp=time.time(), roi=roi)
    writer.write_frame(frame)
    writer.finalize()

    # Save final enhanced output image (for Phase 1 MVP path)
    output_path = os.path.join(writer._root, "enhanced.png")
    upscaled.save(output_path)

    metadata = {
        "processing_time": time.time() - t0,
        "roi_applied": bool(roi),
        "errors": errors,
        "ocr_texts": ocr_texts,
        "plate_texts": plate_texts,
    }
    return PipelineResult(output=output_path, metadata=metadata)
