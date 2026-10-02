"""Lectura de número de camiseta (SPEC-003 RF-1; lector elegido en SPIKE-004: easyocr, confianza >= 0,95)."""

from __future__ import annotations

from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from volley_cv.identity.types import JerseyRead

Box = tuple[float, float, float, float]


class JerseyReader(Protocol):
    def read(self, frame: NDArray[np.uint8], box: Box) -> JerseyRead | None: ...


class EasyOcrJerseyReader:
    """easyocr sobre el torso (12-62 % de la altura de la caja), gris, alto 160 px, solo dígitos.

    `ocr` permite inyectar un objeto con `readtext` (tests); por defecto, `easyocr.Reader` (extra ml).
    """

    def __init__(self, min_conf: float = 0.95, ocr: Any | None = None) -> None:
        self.min_conf = min_conf
        self._ocr = ocr

    def read(self, frame: NDArray[np.uint8], box: Box) -> JerseyRead | None:
        x1, y1, x2, y2 = box
        bh, bw = y2 - y1, x2 - x1
        fh, fw = frame.shape[:2]
        crop = frame[
            max(int(y1 + 0.12 * bh), 0) : min(int(y1 + 0.62 * bh), fh),
            max(int(x1 - 0.05 * bw), 0) : min(int(x2 + 0.05 * bw), fw),
        ]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        scale = 160 / crop.shape[0]
        gray = cv2.cvtColor(
            cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY
        )
        try:
            results = self._reader().readtext(gray, allowlist="0123456789", text_threshold=0.4, low_text=0.3)
        except Exception:  # un fallo del OCR no debe tirar el pipeline: la observación sigue sin número
            return None
        best: JerseyRead | None = None
        for _, text, conf in results:
            t = str(text).strip()
            valid = t.isdigit() and 1 <= len(t) <= 2 and float(conf) >= self.min_conf
            if valid and (best is None or float(conf) > best.confidence):
                best = JerseyRead(int(t), float(conf))
        return best

    def _reader(self) -> Any:
        if self._ocr is None:
            import easyocr  # type: ignore[import-untyped]

            self._ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
        return self._ocr
