"""Filtro de personas en cancha por color del piso (SPEC-002 RF-2).

Una persona está en cancha si su punto de apoyo (centro inferior de la caja) cae sobre la máscara de color
de la cancha dilatada `margin_px` (los pies tapan el piso justo debajo, de ahí el margen). Las regiones
excluidas (p. ej. el marcador sobreimpreso) nunca cuentan.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from volley_cv.video_config import CourtConfig

Box = tuple[float, float, float, float]


class CourtMask:
    def __init__(self, court: CourtConfig, exclude_regions: Sequence[Box] = ()) -> None:
        self.court = court
        self.exclude_regions = list(exclude_regions)
        m = court.margin_px
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1)) if m > 0 else None
        self._close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    def mask(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out = np.zeros(frame.shape[:2], dtype=np.uint8)
        for r in self.court.hsv_ranges:
            out |= np.asarray(
                cv2.inRange(hsv, np.array(r.lower, np.uint8), np.array(r.upper, np.uint8)), dtype=np.uint8
            )
        h, w = out.shape
        excluded = np.zeros_like(out)
        for x1, y1, x2, y2 in self.exclude_regions:
            excluded[int(y1 * h) : int(np.ceil(y2 * h)), int(x1 * w) : int(np.ceil(x2 * w))] = 1
        out[excluded == 1] = 0
        # las líneas blancas (ataque, centro) parten el piso en zonas: se cierran antes de la componente
        out = np.asarray(cv2.morphologyEx(out, cv2.MORPH_CLOSE, self._close), dtype=np.uint8)
        # la cancha es la mancha conexa más grande: se descartan público, sillas, carteles del mismo color
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            out = np.where(labels == largest, 255, 0).astype(np.uint8)
        if self._kernel is not None:
            out = np.asarray(cv2.dilate(out, self._kernel), dtype=np.uint8)
        out[excluded == 1] = 0
        return out

    def in_court(self, frame: NDArray[np.uint8], boxes: Sequence[Box]) -> list[bool]:
        if not boxes:
            return []
        m = self.mask(frame)
        h, w = m.shape
        result = []
        for x1, _, x2, y2 in boxes:
            fx, fy = round((x1 + x2) / 2), round(y2)
            if fy == h:  # pies justo en el borde inferior de la imagen
                fy = h - 1
            result.append(0 <= fx < w and 0 <= fy < h and bool(m[fy, fx]))
        return result
