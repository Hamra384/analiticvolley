"""Regresiones de la revisión independiente de S4 (hallazgos 6, 8, 9, 10, 11)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from volley_cv.__main__ import main
from volley_cv.court import CourtMask
from volley_cv.output.schema import FrameOutput, PlayerOut
from volley_cv.pipeline import Pipeline
from volley_cv.team import TeamClassifier
from volley_cv.video.shots import ShotDetector
from volley_cv.viz import DebugOptions, DebugRenderer

from .test_pipeline import COURT, CUT_AT, TEAMS, FakeDetector, FakeEmbedder, FakeTracker, make_frame


class Spy:
    def __init__(self, log: list[str], name: str, inner: object) -> None:
        self._log, self._name, self._inner = log, name, inner

    def reset(self) -> None:
        self._log.append(f"reset:{self._name}")
        self._inner.reset()  # type: ignore[attr-defined]

    def __getattr__(self, item: str) -> object:
        return getattr(self._inner, item)


class LoggingDetector(FakeDetector):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self._log = log

    def detect(self, frame: np.ndarray) -> np.ndarray:
        self._log.append(f"detect:{self.i}")
        return super().detect(frame)


def test_finding6_all_components_reset_at_cut_before_processing_the_frame(tmp_path: Path) -> None:
    log: list[str] = []
    from volley_cv.identity import IdentityManager

    pipe = Pipeline(
        detector=LoggingDetector(log),
        tracker=Spy(log, "tracker", FakeTracker()),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        court=CourtMask(COURT),
        teams=Spy(log, "teams", TeamClassifier(TEAMS)),  # type: ignore[arg-type]
        manager=Spy(log, "manager", IdentityManager()),  # type: ignore[arg-type]
        renderer=Spy(log, "renderer", DebugRenderer()),  # type: ignore[arg-type]
    )
    pipe.run(((i, make_frame(i)) for i in range(CUT_AT + 3)), tmp_path, write_video=False)
    i = log.index(f"detect:{CUT_AT}")
    assert set(log[i - 4 : i]) == {"reset:tracker", "reset:teams", "reset:manager", "reset:renderer"}
    assert sum(entry.startswith("reset:") for entry in log) == 4


def test_finding8_cut_right_after_first_frame_is_detected() -> None:
    rng = np.random.default_rng(0)
    a = np.kron(rng.integers(0, 255, (18, 32, 3), dtype=np.uint8), np.ones((20, 20, 1), np.uint8))
    b = np.kron(rng.integers(0, 255, (18, 32, 3), dtype=np.uint8), np.ones((20, 20, 1), np.uint8))
    det = ShotDetector()
    assert [det.update(f) for f in (a, b, b)] == [False, True, False]


def test_finding8_pipeline_feeds_first_frame_to_shot_detector(tmp_path: Path) -> None:
    frames = [(0, make_frame(CUT_AT + 1)), (1, make_frame(0)), (2, make_frame(1))]  # corte en el índice 1
    pipe = Pipeline(FakeDetector(), FakeTracker(), FakeEmbedder(), CourtMask(COURT), TeamClassifier(TEAMS))
    assert pipe.run(iter(frames), tmp_path, write_video=False).cuts == [1]


def test_finding9_video_writer_is_closed_on_error(tmp_path: Path) -> None:
    class Boom(FakeDetector):
        def detect(self, frame: np.ndarray) -> np.ndarray:
            if self.i == 5:
                raise RuntimeError("falla del detector")
            return super().detect(frame)

    pipe = Pipeline(Boom(), FakeTracker(), FakeEmbedder(), CourtMask(COURT), TeamClassifier(TEAMS))
    with pytest.raises(RuntimeError):
        pipe.run(((i, make_frame(i)) for i in range(10)), tmp_path, write_video=True)
    import cv2

    cap = cv2.VideoCapture(str(tmp_path / "debug.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5  # archivo cerrado y legible
    cap.release()


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (["--video", "D:/abs/v.mp4", "--video-config", "jpn_arg_2026", "--end", "0:05"], "relativa"),
        (["--video", "../fuera.mp4", "--video-config", "jpn_arg_2026", "--end", "0:05"], "relativa"),
        (
            ["--video", "videos/v.mp4", "--video-config", "jpn_arg_2026", "--start", "0:05", "--end", "0:05"],
            "inicio",
        ),
        (["--clip", "A2", "--weights", "../w.pt"], "relativa"),
    ],
)
def test_finding10_cli_rejects_unsafe_paths_and_empty_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    msg: str,
) -> None:
    monkeypatch.setenv("VOLLEY_DATA_DIR", str(tmp_path))
    assert main(["run", *args]) == 2
    assert msg in capsys.readouterr().err


def _out(frame: int, x: float | None) -> FrameOutput:
    players = []
    if x is not None:
        players.append(
            PlayerOut(
                player_id="TEAM_A_PLAYER_01",
                track_id="track_1",
                team_id="TEAM_A",
                bbox=(x, 100.0, x + 40.0, 300.0),
                confidence=0.9,
                state="TRACKED",
            )
        )
    return FrameOutput(frame=frame, players=players)


def test_finding11_trail_breaks_when_player_disappears() -> None:
    r = DebugRenderer(
        DebugOptions(boxes=False, labels=False, hud=False, state=False, jersey=False, trails=True)
    )
    blank = np.zeros((400, 640, 3), dtype=np.uint8)
    r.draw(blank, _out(0, 50.0))
    r.draw(blank, _out(1, None))  # desaparece
    img = r.draw(blank, _out(2, 450.0))  # reaparece lejos (re-identificado)
    assert not img[300, 100:440].any()  # sin línea que cruce la cancha
    img = r.draw(blank, _out(3, 470.0))
    assert img[300, 475:485].any()  # la trayectoria nueva sí se dibuja
