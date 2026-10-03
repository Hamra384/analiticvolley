"""SPEC-005 RF-1: una sola pasada de YOLO COCO separa personas (clase 0) y pelota (clase 32). Modelo falso."""

from __future__ import annotations

from typing import Any

import numpy as np

from volley_cv.adapters import YoloBallDetector, YoloPersonDetector


class _T:
    def __init__(self, a: list[Any]) -> None:
        self.a = np.array(a, dtype=np.float32)

    def cpu(self) -> _T:
        return self

    def numpy(self) -> np.ndarray:
        return self.a


class _Boxes:
    def __init__(self, rows: list[tuple[float, float, float, float, float, int]]) -> None:
        self.xyxy = _T([r[:4] for r in rows])
        self.conf = _T([r[4] for r in rows])
        self.cls = _T([r[5] for r in rows])


class FakeYolo:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame: np.ndarray, **kw: Any) -> list[Any]:
        self.calls += 1
        rows = [(10, 10, 50, 150, 0.9, 0), (100, 100, 116, 116, 0.3, 32), (5, 5, 20, 40, 0.1, 0)]
        r = type("R", (), {"boxes": _Boxes(rows)})()
        return [r]


def test_person_and_ball_share_one_inference() -> None:
    model = FakeYolo()
    people = YoloPersonDetector(model=model, conf=0.25)
    ball = YoloBallDetector.from_person_detector(people)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    p = people.detect(frame)
    b = ball.detect(frame)
    assert model.calls == 1
    assert p.shape == (1, 5) and p[0, 4] == np.float32(0.9)  # la persona de 0,1 queda bajo el umbral
    np.testing.assert_allclose(b, [[108, 108, 16, 16, 0.3]])  # cx, cy, w, h, conf


def test_ball_detector_runs_its_own_model_when_frame_is_new() -> None:
    model = FakeYolo()
    people = YoloPersonDetector(model=model)
    ball = YoloBallDetector.from_person_detector(people)
    ball.detect(np.zeros((10, 10, 3), dtype=np.uint8))  # sin pasada previa de personas para este frame
    assert model.calls == 1
