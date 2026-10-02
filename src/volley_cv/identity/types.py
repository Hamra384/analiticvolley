"""Tipos de entrada del IdentityManager (SPEC-001)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class Team(StrEnum):
    A = "A"
    B = "B"


class PlayerState(StrEnum):
    DETECTED = "DETECTED"
    TRACKED = "TRACKED"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    REIDENTIFIED = "REIDENTIFIED"


@dataclass(frozen=True)
class JerseyRead:
    """Una lectura de dorsal de un frame (OCR/clasificador), con su confianza en [0, 1]."""

    number: int
    confidence: float


@dataclass(frozen=True, eq=False)
class Observation:
    """Una detección de persona ya asociada a un track de corto plazo del tracker.

    bbox en píxeles (x1, y1, x2, y2). ``team`` lo aporta el clasificador de equipo (None = desconocido).
    ``embedding`` es el descriptor de apariencia (se normaliza internamente).
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    team: Team | None
    embedding: NDArray[np.float32] | None = field(default=None)
    jersey: JerseyRead | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
