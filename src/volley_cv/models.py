"""Interfaces de los componentes con modelos (SPEC-002 RF-8). Las implementaciones reales importan torch,
ultralytics y boxmot de forma diferida (extra `ml`); los tests usan implementaciones falsas."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Box = tuple[float, float, float, float]


class PlayerDetector(Protocol):
    def detect(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Personas detectadas: array (N, 5) con x1, y1, x2, y2, confianza."""
        ...


class Tracker(Protocol):
    def update(self, detections: NDArray[np.float32], frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Tracks del frame: array (M, 6) con x1, y1, x2, y2, track_id, confianza."""
        ...

    def reset(self) -> None: ...


class Embedder(Protocol):
    def embed(self, frame: NDArray[np.uint8], boxes: Sequence[Box]) -> list[NDArray[np.float32] | None]:
        """Un descriptor de apariencia por caja (None si el recorte es vacío)."""
        ...
