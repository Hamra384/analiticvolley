"""SPEC-005 AC-7: el pipeline agrega la pelota a cada frame (detector falso)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from volley_cv.court import CourtMask
from volley_cv.pipeline import Pipeline
from volley_cv.team import TeamClassifier

from .test_pipeline import COURT, CUT_AT, TEAMS, FakeDetector, FakeEmbedder, FakeTracker, N, make_frame


class FakeBallDetector:
    """Pelota en x = 100 + 8 f, y = 150; invisible en los frames 20-24 (tapada)."""

    def __init__(self) -> None:
        self.f = -1

    def detect(self, frame: np.ndarray) -> np.ndarray:
        self.f += 1
        if 20 <= self.f < 25:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array([[100.0 + 8 * self.f, 150.0, 10.0, 10.0, 0.7]], dtype=np.float32)


def run(tmp_path: Path, write_video: bool = False) -> list[dict]:
    pipe = Pipeline(
        FakeDetector(),
        FakeTracker(),
        FakeEmbedder(),
        CourtMask(COURT),
        TeamClassifier(TEAMS),
        ball_detector=FakeBallDetector(),
    )
    pipe.run(((i, make_frame(i)) for i in range(N)), tmp_path, write_video=write_video)
    return [json.loads(x) for x in (tmp_path / "frames.jsonl").read_text(encoding="utf-8").splitlines()]


def test_ac7_every_frame_has_a_ball_state(tmp_path: Path) -> None:
    rows = run(tmp_path, write_video=True)
    assert all(r["ball"] is not None for r in rows)
    states = [r["ball"]["state"] for r in rows]
    assert states[0] == "DETECTED"
    assert set(states[20:25]) == {"PREDICTED"}  # tapada: predicha, nunca detectada
    assert states[25] == "TRACKED"
    assert (tmp_path / "debug.mp4").stat().st_size > 0


def test_ac7_ball_track_restarts_after_a_cut(tmp_path: Path) -> None:
    rows = run(tmp_path)
    assert rows[CUT_AT]["ball"]["state"] == "DETECTED"
    assert rows[CUT_AT]["ball"]["track_id"] != rows[CUT_AT - 1]["ball"]["track_id"]


def test_without_ball_detector_ball_is_null(tmp_path: Path) -> None:
    pipe = Pipeline(FakeDetector(), FakeTracker(), FakeEmbedder(), CourtMask(COURT), TeamClassifier(TEAMS))
    pipe.run(((i, make_frame(i)) for i in range(5)), tmp_path, write_video=False)
    rows = [json.loads(x) for x in (tmp_path / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["ball"] is None for r in rows)
