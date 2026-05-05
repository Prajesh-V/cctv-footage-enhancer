from .frame import Frame
from .cancellation import CancellationToken
from .interfaces import PipelineResult
from .output_writer import OutputWriter
import time
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


from typing import Optional, Dict
import os
import cv2
try:
    from ml_runtime import MLRuntime
except ImportError:
    from ..ml_runtime import MLRuntime


def _normalize_roi(roi: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not roi:
        return None
    try:
        x = float(roi.get("x", 0.0))
        y = float(roi.get("y", 0.0))
        w = float(roi.get("w", 1.0))
        h = float(roi.get("h", 1.0))
    except Exception:
        return None
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}

def process_image_pipeline(
    input_path: str,
    roi: Optional[Dict[str, float]],
    upscale_factor: int,
    enable_face_restore: bool,
    plate_detection: bool,
    cancellation_token: CancellationToken,
    writer: OutputWriter,
) -> PipelineResult:
    """Real ML-powered image processing pipeline.

    Integrates RealESRGAN, GFPGAN, and YOLO/OCR via MLRuntime to provide
    high-quality enhancement and forensic analysis.
    """
    t0 = time.time()
    errors = []
    
    # 1. Initialize ML Runtime
    use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
    try:
        ml_runtime = MLRuntime(use_gpu=use_gpu)
    except Exception as e:
        errors.append(f"MLRuntime Init Error: {str(e)}")
        return PipelineResult(output=None, metadata={"processing_time": time.time() - t0, "errors": errors})

    # 2. Load Image
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Failed to load image via OpenCV")
    except Exception as e:
        errors.append(f"Image Load Error: {str(e)}")
        return PipelineResult(output=None, metadata={"processing_time": time.time() - t0, "errors": errors})

    # 3. Normalize ROI
    norm_roi = _normalize_roi(roi)

    # 4. Check Cancellation
    if cancellation_token.cancelled:
        return PipelineResult(output=None, metadata={"processing_time": time.time() - t0, "errors": ["cancelled"]})

    # 5. Process Frame
    try:
        # Note: MLRuntime.process_frame handles upscaling, face restoration, and plate detection
        enhanced_img, ml_results = ml_runtime.process_frame(
            img, 
            norm_roi, 
            enable_face_restore=enable_face_restore, 
            detect_plates=plate_detection
        )
    except Exception as e:
        errors.append(f"Processing Error: {str(e)}")
        return PipelineResult(output=None, metadata={"processing_time": time.time() - t0, "errors": errors})

    # 6. Save Result
    try:
        # Use the writer to handle output storage
        frame = Frame(data=enhanced_img, index=0, timestamp=time.time(), roi=norm_roi)
        writer.write_frame(frame)
        writer.finalize()
        
        # Determine final output path
        output_path = getattr(writer, 'path', None)
        if not output_path:
            # Fallback if path isn't directly exposed
            root = getattr(writer, '_root', 'clarityai/outputs/streams/default')
            output_path = os.path.join(root, "enhanced_image.png")
            cv2.imwrite(output_path, enhanced_img)
    except Exception as e:
        errors.append(f"Save Error: {str(e)}")
        output_path = None

    metadata = {
        "processing_time": time.time() - t0,
        "roi_applied": bool(norm_roi),
        "errors": errors,
        "face_restored": ml_results.get("face_restored", False),
        "plates": ml_results.get("plates", []),
        "ocr_texts": ml_results.get("ocr_texts", []),
    }
    
    return PipelineResult(output=output_path, metadata=metadata)
from .execution import submit
__all__ = [
    "Frame",
    "CancellationToken",
    "PipelineResult",
    "OutputWriter",
    "submit",
]
