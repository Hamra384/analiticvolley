"""Descriptor de apariencia por histograma de color HSV (SPEC-002 RF-4; decisión por SPIKE-003).

En A2/K2 separó mejor a jugadores distintos (incluidos compañeros con la misma camiseta) que ResNet18
ImageNet y que OSNet x0.25 MSMT17, sin GPU y sin pesos con licencia no comercial. Histograma 12x6x6
(H, S, V) del recorte completo, raíz cuadrada (Hellinger) y norma 1, para que la distancia coseno sea
comparable con los umbrales.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

Box = tuple[float, float, float, float]
BINS = [12, 6, 6]
RANGES = [0, 180, 0, 256, 0, 256]


class ColorHistEmbedder:
    def embed(self, frame: NDArray[np.uint8], boxes: Sequence[Box]) -> list[NDArray[np.float32] | None]:
        h, w = frame.shape[:2]
        out: list[NDArray[np.float32] | None] = []
        for x1, y1, x2, y2 in boxes:
            crop = frame[max(int(y1), 0) : min(int(y2), h), max(int(x1), 0) : min(int(x2), w)]
            if crop.size == 0:
                out.append(None)
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = np.asarray(cv2.calcHist([hsv], [0, 1, 2], None, BINS, RANGES), dtype=np.float64).ravel()
            v = np.sqrt(hist / max(float(hist.sum()), 1.0))
            norm = float(np.linalg.norm(v))
            out.append((v / norm).astype(np.float32) if norm > 0 else None)
        return out
