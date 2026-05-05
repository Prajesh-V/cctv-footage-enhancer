from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class Frame:
    """Simple frame abstraction for video processing.

    data: 2D/3D numpy array representing the image frame (H x W x C).
    index: Frame index in the video sequence.
    timestamp: Frame timestamp (seconds since epoch or relative time).
    roi: Optional ROI descriptor for this frame (normalized coordinates or pixel coords).
    metadata: Optional metadata dict for passing processing options.
    """

    data: np.ndarray
    index: int
    timestamp: float
    roi: Optional[Dict] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
