"""Utilidades compartidas por los spikes (código experimental, no de producción).

Rutas resueltas con volley_cv.config: los videos, pesos y cachés viven en el directorio de datos.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from volley_cv.config import PROJECT_ROOT, Clip, load_clips, load_settings

try:  # usar el almacén de certificados del SO (la red local intercepta TLS); la verificación sigue activa
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

CATALOG = PROJECT_ROOT / "configs" / "eval" / "clips.yaml"


@dataclass(frozen=True)
class ClipInfo:
    clip: Clip
    path: Path
    fps: float
    width: int
    height: int
    start_frame: int
    n_frames: int


def data_dir() -> Path:
    return load_settings().data_dir


def cache_dir(*parts: str) -> Path:
    d = data_dir().joinpath("cache", *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def clips(ids: list[str] | None = None) -> list[ClipInfo]:
    catalog = load_clips(CATALOG)
    out = []
    for c in catalog.clips:
        if ids and c.id not in ids:
            continue
        path = catalog.video_path(c, data_dir())
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        info = ClipInfo(
            clip=c,
            path=path,
            fps=fps,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            start_frame=round(c.start_s * fps),
            n_frames=round(c.duration_s * fps),
        )
        cap.release()
        out.append(info)
    return out


def frames(info: ClipInfo) -> Iterator[tuple[int, np.ndarray]]:
    """Frames del clip, indexados desde 0 (relativos al inicio del clip)."""
    cap = cv2.VideoCapture(str(info.path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, info.start_frame)
    try:
        for i in range(info.n_frames):
            ok, frame = cap.read()
            if not ok:
                return
            yield i, frame
    finally:
        cap.release()
