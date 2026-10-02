"""Detección de cortes de edición en streaming (SPEC-002 RF-1). Calibrado en la auditoría S2: en los videos
de evaluación, ratio 6 y diferencia mínima 3 separan cortes (94-97 % en segundos exactos) de paneos."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from numpy.typing import NDArray


class ShotDetector:
    def __init__(self, ratio: float = 6.0, min_diff: float = 3.0, window: int = 61) -> None:
        self.ratio, self.min_diff, self.window = ratio, min_diff, window
        self._prev: NDArray[np.int16] | None = None
        self._history: deque[float] = deque(maxlen=window)

    def update(self, frame: NDArray[np.uint8]) -> bool:
        """True si `frame` empieza una toma nueva (corte respecto del frame anterior)."""
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        prev, self._prev = self._prev, gray
        if prev is None:
            return False
        diff = float(np.abs(gray - prev).mean())
        baseline = float(np.median(self._history)) if self._history else diff
        self._history.append(diff)
        return diff > self.min_diff and diff > self.ratio * max(baseline, 0.5)

    def reset(self) -> None:
        self._prev = None
        self._history.clear()
