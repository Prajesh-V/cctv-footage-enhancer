from __future__ import annotations

import os
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

# ROOT_DIR points to the 'clarityai' project root (parent of 'backend')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

_REALESRGAN = None
_GFPGAN = None
_YOLO = None
_EASYOCR = None
_TORCH = None
_NVML = None


def _import_realesrgan():
    global _REALESRGAN
    if _REALESRGAN is None:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            _REALESRGAN = {'RRDBNet': RRDBNet, 'RealESRGANer': RealESRGANer}
        except Exception as e:
            raise ImportError('RealESRGAN not installed: ' + str(e))
    return _REALESRGAN


def _import_gfpgan():
    global _GFPGAN
    if _GFPGAN is None:
        try:
            from gfpgan import GFPGANer
            _GFPGAN = {'GFPGANer': GFPGANer}
        except Exception as e:
            raise ImportError('GFPGAN not installed: ' + str(e))
    return _GFPGAN


def _import_yolo():
    global _YOLO
    if _YOLO is None:
        try:
            from ultralytics import YOLO
            _YOLO = {'YOLO': YOLO}
        except Exception as e:
            raise ImportError('YOLOv8 not installed: ' + str(e))
    return _YOLO


def _import_easyocr():
    global _EASYOCR
    if _EASYOCR is None:
        try:
            import easyocr
            _EASYOCR = {'Reader': easyocr.Reader}
        except Exception as e:
            raise ImportError('EasyOCR not installed: ' + str(e))
    return _EASYOCR


def _import_torch():
    global _TORCH
    if _TORCH is None:
        try:
            import torch
            _TORCH = torch
        except Exception as e:
            raise ImportError('PyTorch not installed: ' + str(e))
    return _TORCH


def _import_nvml():
    global _NVML
    if _NVML is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML = pynvml
        except Exception:
            _NVML = None
    return _NVML


@dataclass
class VRAMInfo:
    used_mb: int
    total_mb: int
    free_mb: int


class VRAMMonitor:
    def __init__(self, max_vram_gb: float = 7.0):
        self.max_vram_bytes = int(max_vram_gb * 1024 * 1024 * 1024)
        self.nvml = _import_nvml()
        self._device_index = 0

    def get_vram_info(self) -> Optional[VRAMInfo]:
        if self.nvml is None:
            return None
        try:
            handle = self.nvml.nvmlDeviceGetHandleByIndex(self._device_index)
            info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            return VRAMInfo(
                used_mb=int(info.used / 1024 / 1024),
                total_mb=int(info.total / 1024 / 1024),
                free_mb=int(info.free / 1024 / 1024),
            )
        except Exception:
            return None

    def check_vram_available(self) -> bool:
        info = self.get_vram_info()
        if info is None:
            return True
        return info.used_mb * 1024 * 1024 < self.max_vram_bytes


class RealESRGANUpscaler:
    def __init__(self, use_gpu: bool = False, upscale_factor: int = 2, max_input_size: Tuple[int, int] = (1920, 1080)):
        self.use_gpu = use_gpu
        self.upscale_factor = upscale_factor
        self.max_input_size = max_input_size
        self.model = None
        self._device = None

    def _init_model(self):
        if self.model is not None:
            return
        torch = _import_torch()
        realesrgan = _import_realesrgan()
        if self.use_gpu and torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        # Try to find the weight file in project root OR backend folder
        model_path = os.path.join(ROOT_DIR, "RealESRGAN_x4plus.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(ROOT_DIR, "backend", "RealESRGAN_x4plus.pth")
        
        if not os.path.exists(model_path):
            print(f"CRITICAL ERROR: {model_path} not found!")
            print("Please download RealESRGAN_x4plus.pth and place it in the backend folder.")
            raise FileNotFoundError(f"Model weights missing: {model_path}")

        # Explicitly define the model architecture (RRDBNet) for RealESRGAN_x4plus
        from basicsr.archs.rrdbnet_arch import RRDBNet
        rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        model = realesrgan["RealESRGANer"](
            scale=4,
            model_path=model_path,
            model=rrdb_model,
            device=self._device,
            half=self.use_gpu,
        )
        self.model = model

    def upscale(self, img: np.ndarray) -> np.ndarray:
        self._init_model()
        h, w = img.shape[:2]
        if w > self.max_input_size[0] or h > self.max_input_size[1]:
            scale = min(self.max_input_size[0] / w, self.max_input_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            import cv2
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = self.model.enhance(img, outscale=self.upscale_factor)
        if isinstance(result, tuple):
            return result[0]
        return result


class GFPGANRestorer:
    def __init__(self, use_gpu: bool = False, fidelity_weight: float = 0.5):
        self.use_gpu = use_gpu
        self.fidelity_weight = fidelity_weight
        self.model = None
        self._device = None

    def _init_model(self):
        if self.model is not None:
            return
        torch = _import_torch()
        gfpgan = _import_gfpgan()
        if self.use_gpu and torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        model_path = os.path.join(ROOT_DIR, "GFPGANv1.4.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(ROOT_DIR, "backend", "GFPGANv1.4.pth")
            
        if not os.path.exists(model_path):
            print(f"CRITICAL ERROR: {model_path} not found!")
            print("Please download GFPGANv1.4.pth and place it in the backend folder.")
            raise FileNotFoundError(f"Model weights missing: {model_path}")

        self.model = gfpgan['GFPGANer'](
            model_path=model_path,
            upscale=1, # We already upscale using RealESRGAN, so GFPGAN should only restore faces
            arch='clean',
            channel_multiplier=2,
            device=self._device,
        )

    def restore_faces(self, img: np.ndarray) -> np.ndarray:
        self._init_model()
        _, _, restored_img = self.model.enhance(img, has_aligned=False, paste_back=True)
        return restored_img if restored_img is not None else img


class YOLOPlateDetector:
    def __init__(self, use_gpu: bool = False, conf_threshold: float = 0.6):
        self.use_gpu = use_gpu
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_name = 'yolov8n.pt'

    def _init_model(self):
        if self.model is not None:
            return
        yolo = _import_yolo()
        self.model = yolo['YOLO'](self.model_name)
        if self.use_gpu:
            self.model.to('cuda')

    def detect_plates(self, img: np.ndarray) -> List[Dict]:
        self._init_model()
        results = self.model(img, conf=self.conf_threshold)
        plates = []
        for r in results:
            for box in r.boxes:
                # Explicitly convert to standard Python int/float to avoid JSON serialization errors
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                plates.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)], 
                    'confidence': float(box.conf[0])
                })
        return plates


class EasyOCRProcessor:
    def __init__(self, use_gpu: bool = False, conf_threshold: float = 0.6):
        self.use_gpu = use_gpu and _import_torch() is not None and _import_torch().cuda.is_available()
        self.conf_threshold = conf_threshold
        self.reader = None

    def _init_reader(self):
        if self.reader is not None:
            return
        easyocr = _import_easyocr()
        self.reader = easyocr['Reader'](['en'], gpu=self.use_gpu)

    def preprocess_for_ocr(self, plate_img: np.ndarray) -> np.ndarray:
        import cv2
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def read_text(self, plate_img: np.ndarray) -> List[Dict]:
        self._init_reader()
        preprocessed = self.preprocess_for_ocr(plate_img)
        results = self.reader.readtext(preprocessed)
        texts = []
        for bbox, text, conf in results:
            if conf >= self.conf_threshold:
                # Convert bbox coordinates to standard Python ints
                clean_bbox = [[int(p[0]), int(p[1])] for p in bbox]
                texts.append({'text': text, 'confidence': float(conf), 'bbox': clean_bbox})
        return texts


class MLRuntime:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.vram_monitor = VRAMMonitor() if use_gpu else None
        self.upscaler = RealESRGANUpscaler(use_gpu=use_gpu)
        self.face_restorer = GFPGANRestorer(use_gpu=use_gpu)
        self.plate_detector = YOLOPlateDetector(use_gpu=use_gpu)
        self.ocr_processor = EasyOCRProcessor(use_gpu=use_gpu)

    def process_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Dict] = None,
        enable_face_restore: bool = True,
        detect_plates: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        t0 = time.time()
        result = {'face_restored': False, 'plates': [], 'ocr_texts': [], 'processing_time': 0.0}

        if self.use_gpu and self.vram_monitor and not self.vram_monitor.check_vram_available():
            raise RuntimeError('VRAM limit exceeded')

        h, w = frame.shape[:2]
        if roi:
            x1 = int(roi.get('x', 0) * w)
            y1 = int(roi.get('y', 0) * h)
            x2 = int((roi.get('x', 0) + roi.get('w', 1)) * w)
            y2 = int((roi.get('y', 0) + roi.get('h', 1)) * h)
            roi_frame = frame[y1:y2, x1:x2]
        else:
            roi_frame = frame

        upscaled = self.upscaler.upscale(roi_frame)

        if enable_face_restore:
            try:
                upscaled = self.face_restorer.restore_faces(upscaled)
                result['face_restored'] = True
            except Exception:
                pass

        plates = []
        ocr_texts = []
        if detect_plates:
            plates = self.plate_detector.detect_plates(upscaled)
            for plate in plates:
                bx1, by1, bx2, by2 = plate['bbox']
                plate_img = upscaled[by1:by2, bx1:bx2]
                texts = self.ocr_processor.read_text(plate_img)
                ocr_texts.extend(texts)
            result['plates'] = plates
            result['ocr_texts'] = ocr_texts

        final = upscaled

        result['processing_time'] = time.time() - t0
        return final, result
