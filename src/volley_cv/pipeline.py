"""Pipeline de jugadores: video → cortes → detección → cancha → tracking → equipo + apariencia → identidad →
salida JSONL (+ video de debug) (SPEC-002)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from volley_cv.court import CourtMask, feet_in
from volley_cv.identity import IdentityManager, Observation
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
    ) -> None:
        self.play_margin = play_margin
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
        teams = self.teams.classify(frame, boxes)
        embs = self.embedder.embed(frame, boxes)
        return [
            Observation(
                track_id=int(t[4]),
                bbox=box,
                confidence=float(min(max(t[5], 0.0), 1.0)),
                team=team,
                embedding=emb,
                in_court=inside,
            )
            for t, box, team, emb, inside in zip(tracks, boxes, teams, embs, in_court, strict=True)
        ]
