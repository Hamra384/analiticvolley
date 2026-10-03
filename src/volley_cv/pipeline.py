"""Pipeline de jugadores: video → cortes → detección → cancha → tracking → equipo + apariencia → identidad →
salida JSONL (+ video de debug) (SPEC-002)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from volley_cv.court import CourtMask, feet_in
from volley_cv.identity import IdentityManager, JerseyRead, Observation
from volley_cv.jersey_reader import JerseyReader
from volley_cv.merges import apply_merges
from volley_cv.models import Embedder, PlayerDetector, Tracker
from volley_cv.team import TeamClassifier
from volley_cv.video.shots import ShotDetector
from volley_cv.viz import DebugRenderer

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    frames: int
    cuts: list[int]
    fps: float
    jsonl: Path
    video: Path | None


class Pipeline:
    def __init__(
        self,
        detector: PlayerDetector,
        tracker: Tracker,
        embedder: Embedder,
        court: CourtMask,
        teams: TeamClassifier,
        manager: IdentityManager | None = None,
        shots: ShotDetector | None = None,
        renderer: DebugRenderer | None = None,
        play_margin: float = 0.12,
        jersey_reader: JerseyReader | None = None,
        read_min_height: float = 0.18,
        read_stride: int = 5,
        read_max_occlusion: float = 0.15,
        forget_iou: float = 0.3,
    ) -> None:
        self.play_margin = play_margin
        self.jersey_reader = jersey_reader
        self.read_min_height, self.read_stride = read_min_height, read_stride
        self.read_max_occlusion = read_max_occlusion
        self.forget_iou = forget_iou
        self._frame_i = 0
        self._last_read: dict[int, int] = {}  # track_id -> último frame leído (SPEC-003 RF-2)
        self._track_number: dict[int, int] = {}  # track_id -> último número leído (SPEC-003 RF-6b)
        self.detector, self.tracker, self.embedder = detector, tracker, embedder
        self.court, self.teams = court, teams
        self.manager = manager or IdentityManager()
        self.shots = shots or ShotDetector()
        self.renderer = renderer or DebugRenderer()

    def run(
        self,
        frames: Iterable[tuple[int, NDArray[np.uint8]]],
        out_dir: Path,
        fps: float = 30.0,
        write_video: bool = True,
    ) -> RunSummary:
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl = out_dir / "frames.jsonl"
        video_path = out_dir / "debug.mp4" if write_video else None
        writer: cv2.VideoWriter | None = None
        cuts: list[int] = []
        n = 0
        t0 = time.perf_counter()
        try:
            with jsonl.open("w", encoding="utf-8") as f:
                for idx, frame in frames:
                    cut = self.shots.update(frame)  # el primer frame nunca es corte
                    if cut:
                        cuts.append(idx)
                        self.manager.reset()
                        self.tracker.reset()
                        self.teams.reset()
                        self.renderer.reset()
                        self._last_read.clear()
                        self._track_number.clear()
                    out = self.manager.update(idx, self._observations(frame))
                    f.write(out.model_dump_json() + "\n")
                    if video_path is not None:
                        if writer is None:
                            h, w = frame.shape[:2]
                            writer = cv2.VideoWriter(
                                str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h)
                            )
                        writer.write(self.renderer.draw(frame, out, cut))
                    n += 1
        finally:
            if writer is not None:
                writer.release()  # el video queda legible aunque el proceso falle a mitad del clip
        # SPEC-003 RF-4: las fusiones por número se aplican a todo el clip (el video de debug muestra las
        # identidades tal como se decidieron en cada momento; el JSONL queda con la identidad final)
        merges = self.manager.merges
        (out_dir / "merges.json").write_text(
            json.dumps([{"from": s, "to": d, "frame": f} for s, d, f in merges], indent=1), encoding="utf-8"
        )
        if merges:
            apply_merges(jsonl, [(s, d) for s, d, _ in merges])
        elapsed = time.perf_counter() - t0
        summary = RunSummary(n, cuts, n / elapsed if elapsed > 0 else 0.0, jsonl, video_path)
        log.info("procesados %d frames a %.1f FPS; cortes: %s", n, summary.fps, cuts)
        return summary

    def _observations(self, frame: NDArray[np.uint8]) -> list[Observation]:
        court_mask, play_mask = self.court.masks(frame, self.play_margin)
        dets = self.detector.detect(frame)
        if len(dets):
            det_boxes = [(float(d[0]), float(d[1]), float(d[2]), float(d[3])) for d in dets]
            # RF-2b: al tracker pasa toda la zona de juego (incluye la zona libre: saque, defensa)
            keep = np.array(feet_in(play_mask, det_boxes), dtype=bool)
            # RF-3b: los oficiales se descartan antes de que puedan ocupar un track
            officials = self.teams.is_official(frame, [b for b, k in zip(det_boxes, keep, strict=True) if k])
            idx = np.flatnonzero(keep)
            keep[idx[np.array(officials, dtype=bool)]] = False
            dets = dets[keep]
        tracks = self.tracker.update(dets.reshape(-1, 5), frame)
        if not len(tracks):
            return []
        boxes = [(float(t[0]), float(t[1]), float(t[2]), float(t[3])) for t in tracks]
        in_court = feet_in(court_mask, boxes)
        reads = self._read_numbers(frame, tracks, boxes)
        roles = self.teams.classify_roles(frame, boxes, numbers=self._team_numbers(tracks, boxes, reads))
        embs = self.embedder.embed(frame, boxes)
        return [
            Observation(
                track_id=int(t[4]),
                bbox=box,
                confidence=float(min(max(t[5], 0.0), 1.0)),
                team=team,
                embedding=emb,
                jersey=read,
                in_court=inside,
                libero=libero,
            )
            for t, box, (team, libero), emb, read, inside in zip(
                tracks, boxes, roles, embs, reads, in_court, strict=True
            )
        ]

    def _team_numbers(
        self,
        tracks: NDArray[np.float32],
        boxes: list[tuple[float, float, float, float]],
        reads: list[JerseyRead | None],
    ) -> list[int | None]:
        """SPEC-003 RF-6b: para el equipo vale el último número leído del track, no solo el del frame (el OCR
        lee pocas veces; entre lecturas el color ambiguo de un líbero lo mandaba al rival). Se olvida en una
        superposición, donde el tracker puede pasar el ID a otra persona (revisión S5a M3)."""
        for k, (t, r) in enumerate(zip(tracks, reads, strict=True)):
            tid = int(t[4])
            if r is not None:
                self._track_number[tid] = r.number
            elif any(_iou(boxes[k], o) > self.forget_iou for j, o in enumerate(boxes) if j != k):
                self._track_number.pop(tid, None)
        return [self._track_number.get(int(t[4])) for t in tracks]

    def _read_numbers(
        self,
        frame: NDArray[np.uint8],
        tracks: NDArray[np.float32],
        boxes: list[tuple[float, float, float, float]],
    ) -> list[JerseyRead | None]:
        """SPEC-003 RF-2: solo cajas grandes y cada `read_stride` frames por track (costo acotado).

        RF-2b: no se lee un torso tapado por alguien más cerca de la cámara (leería otro número).
        """
        self._frame_i += 1
        out: list[JerseyRead | None] = [None] * len(boxes)
        if self.jersey_reader is None:
            return out
        min_h = self.read_min_height * frame.shape[0]
        for k, (t, box) in enumerate(zip(tracks, boxes, strict=True)):
            tid = int(t[4])
            if (
                box[3] - box[1] < min_h
                or self._frame_i - self._last_read.get(tid, -(10**9)) < self.read_stride
                or _torso_occlusion(box, boxes) > self.read_max_occlusion
            ):
                continue
            self._last_read[tid] = self._frame_i
            out[k] = self.jersey_reader.read(frame, box)
        return out


def _torso_occlusion(
    box: tuple[float, float, float, float], boxes: list[tuple[float, float, float, float]]
) -> float:
    """Máxima fracción del torso (12 a 62 % del alto, como el lector) cubierta por una caja con los pies más
    abajo (más cerca de la cámara, por lo tanto delante)."""
    x1, y1, x2, y2 = box
    ty1, ty2 = y1 + 0.12 * (y2 - y1), y1 + 0.62 * (y2 - y1)
    area = (x2 - x1) * (ty2 - ty1)
    worst = 0.0
    for o in boxes:
        if o is box or o[3] <= y2 or area <= 0:
            continue
        w = min(x2, o[2]) - max(x1, o[0])
        h = min(ty2, o[3]) - max(ty1, o[1])
        if w > 0 and h > 0:
            worst = max(worst, w * h / area)
    return worst


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
