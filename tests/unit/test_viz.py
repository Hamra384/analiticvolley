"""AC-7 (SPEC-002): el debug dibuja cajas por equipo con player_id corto; todo desactivado no cambia nada."""

import numpy as np

from volley_cv.output.schema import FrameOutput, PlayerOut
from volley_cv.viz import DebugOptions, DebugRenderer, short_id


def out(frame: int = 0) -> FrameOutput:
    def p(pid: str, team: str, x: float, jersey: int | None) -> PlayerOut:
        return PlayerOut(
            player_id=pid,
            track_id="track_3",
            team_id=team,
            jersey_number=jersey,  # type: ignore[arg-type]
            jersey_confidence=0.9 if jersey else 0.0,
            bbox=(x, 200.0, x + 60.0, 360.0),
            confidence=0.9,
            state="TRACKED",
        )

    return FrameOutput(
        frame=frame,
        players=[p("TEAM_A_PLAYER_04", "TEAM_A", 100, 7), p("TEAM_B_PLAYER_12", "TEAM_B", 400, None)],
    )


def test_short_id() -> None:
    assert short_id("TEAM_A_PLAYER_04") == "A04"
    assert short_id("TEAM_B_PLAYER_12") == "B12"


def test_ac7_draws_boxes_with_team_colors() -> None:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    img = DebugRenderer().draw(blank, out())
    assert not np.array_equal(img, blank)
    # borde izquierdo de cada caja pintado con el color de su equipo, y colores distintos entre equipos
    a = img[280, 100]
    b = img[280, 400]
    assert a.any() and b.any() and not np.array_equal(a, b)
    assert np.array_equal(blank, np.zeros_like(blank))  # no modifica la entrada


def test_ac7_everything_disabled_leaves_image_unchanged() -> None:
    opts = DebugOptions(
        boxes=False, labels=False, track_ids=False, jersey=False, state=False, trails=False, hud=False
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert np.array_equal(DebugRenderer(opts).draw(blank, out()), blank)


def test_ac7_labels_are_drawn_above_box() -> None:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    only_labels = DebugOptions(boxes=False, labels=True, trails=False, hud=False, state=False, jersey=False)
    img = DebugRenderer(only_labels).draw(blank, out())
    assert img[170:200, 100:160].any()  # texto encima de la caja A04
    assert not img[220:340, 110:150].any()  # interior de la caja sin tocar


def test_trails_accumulate_and_reset() -> None:
    r = DebugRenderer(
        DebugOptions(boxes=False, labels=False, hud=False, state=False, jersey=False, trails=True)
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    r.draw(blank, out(0))
    moved = out(1).model_copy(
        update={"players": [out(1).players[0].model_copy(update={"bbox": (160.0, 200.0, 220.0, 360.0)})]}
    )
    img = r.draw(blank, moved)
    assert img[360, 130:190].any()  # línea de trayectoria entre los pies de ambos frames
    r.reset()
    img = r.draw(blank, moved)
    assert not img[360, 130:150].any()


def _ball_frame(state: str, pos: tuple[float, float] | None) -> FrameOutput:
    from volley_cv.output.schema import BallOut

    return FrameOutput(
        frame=0,
        players=[],
        ball=BallOut(track_id="ball_1", position=pos, confidence=0.5, state=state),  # type: ignore[arg-type]
    )


def test_spec005_detected_ball_is_drawn_filled_and_predicted_hollow() -> None:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    det = DebugRenderer(DebugOptions(hud=False)).draw(img, _ball_frame("DETECTED", (100.0, 100.0)))
    pred = DebugRenderer(DebugOptions(hud=False)).draw(img, _ball_frame("PREDICTED", (100.0, 100.0)))
    assert det[100, 100].any()  # relleno en el centro
    assert not pred[100, 100].any() and pred.any()  # solo el contorno


def test_spec005_lost_ball_draws_nothing() -> None:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = DebugRenderer(DebugOptions(hud=False)).draw(img, _ball_frame("LOST", None))
    assert not out.any()


def test_spec005_hud_shows_ball_state() -> None:
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    with_ball = DebugRenderer().draw(img, _ball_frame("PREDICTED", (300.0, 150.0)))
    without = DebugRenderer().draw(img, FrameOutput(frame=0, players=[]))
    assert (with_ball[:28] != without[:28]).any()
