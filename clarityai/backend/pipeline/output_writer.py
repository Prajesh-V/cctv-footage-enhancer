from __future__ import annotations

from abc import ABC, abstractmethod
import os
from PIL import Image
from typing import Optional

from .frame import Frame


class OutputWriter(ABC):
    @abstractmethod
    def write_frame(self, frame: Frame) -> None:
        """Persist or forward a single processed frame."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        """Finalize the output (e.g., flush buffers, close files)."""
        raise NotImplementedError


class DiskOutputWriter(OutputWriter):
    """Concrete OutputWriter that writes per-frame images to disk under a job directory.

    This is intended for MVP streaming output where frames are produced incrementally.
    """

    def __init__(self, job_id: str, output_root: str = "clarityai/outputs/streams") -> None:
        self.job_id = job_id
        self._root = os.path.join(output_root, job_id)
        os.makedirs(self._root, exist_ok=True)

    def write_frame(self, frame: Frame) -> None:
        if frame.data is None:
            return
        # Convert BGR (OpenCV) to RGB (Pillow)
        if len(frame.data.shape) == 3 and frame.data.shape[2] == 3:
            rgb_data = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)
        else:
            rgb_data = frame.data
        img = Image.fromarray(rgb_data)
        path = os.path.join(self._root, f"frame_{frame.index:06d}.png")
        img.save(path)

    def finalize(self) -> None:
        # No-op finalization for MVP; could add packaging (zip) or video muxing later
        pass


try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


class VideoOutputWriter(OutputWriter):
    """Video writer that encodes processed frames to a video file on disk.

    Frames are expected to be numpy arrays (BGR order, as produced by OpenCV).
    The writer is lazily initialized on the first frame to discover frame size.
    """

    def __init__(self, job_id: str, fps: float, width: int = 0, height: int = 0, output_root: str = "clarityai/outputs/videos", progress_callback=None) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) must be installed to use VideoOutputWriter.")
        self.job_id = job_id
        self._root = os.path.join(output_root, job_id)
        os.makedirs(self._root, exist_ok=True)
        self.path = os.path.join(self._root, "enhanced.mp4")
        self._fps = fps if fps > 0 else 30.0
        self._width = int(width)
        self._height = int(height)
        self._writer: Optional[any] = None
        self.progress_callback = progress_callback
        self._frame_count = 0
        self._total_frames: Optional[int] = None

    def _ensure_writer(self, frame_width: int, frame_height: int) -> None:
        if self._writer is None:
            # If width/height weren't provided or were 0, use frame dimensions
            if self._width == 0: self._width = frame_width
            if self._height == 0: self._height = frame_height
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.path, fourcc, self._fps, (self._width, self._height), True)
            
            if not getattr(self._writer, 'isOpened', lambda: False)():
                raise RuntimeError(f"VideoWriter failed to open at {self.path}")

    def write_frame(self, frame: Frame) -> None:
        if frame.data is None:
            return
        
        img = frame.data
        h, w = img.shape[:2]
        self._ensure_writer(w, h)
        
        if img.shape[0] != self._height or img.shape[1] != self._width:
            img = cv2.resize(img, (self._width, self._height))
            
        self._writer.write(img)
        self._frame_count += 1
        
        if self._total_frames and self.progress_callback:
            try:
                self.progress_callback(self._frame_count, self._total_frames)
            except:
                pass

    def finalize(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
