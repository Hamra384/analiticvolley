"""Lectura de un tramo de video con índices de frame absolutos."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from volley_cv.config import ConfigError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ConfigError(f"No se pudo abrir el video: {path}")
    try:
        return VideoInfo(
            path=path,
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frames(path: Path, start_s: float, end_s: float) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """Frames de [start_s, end_s) con su índice absoluto en el video."""
    info = probe(path)
    first, last = round(start_s * info.fps), round(end_s * info.fps)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    try:
        for idx in range(first, last):
            ok, frame = cap.read()
            if not ok:
                log.warning("el video terminó o falló en el frame %d (se procesaron %d)", idx, idx - first)
                return
            yield idx, np.asarray(frame, dtype=np.uint8)
    finally:
        cap.release()
