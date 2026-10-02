"""SPEC-004 AC-1…AC-7: plantel cerrado por equipo (IDs 01-06 + líbero 07, re-identificación por descarte)."""

from __future__ import annotations

import numpy as np

from volley_cv.identity import IdentityConfig, IdentityManager, JerseyRead, Observation, Team
from volley_cv.output.schema import FrameOutput

E = np.eye(32, dtype=np.float32)
CFG = IdentityConfig()
XS = [150.0, 450.0, 750.0, 1050.0, 1350.0, 1650.0]


def emb(k: int, drift: int | None = None) -> np.ndarray:
    """Apariencia del jugador k; con `drift`, cambiada (luz, pose): coseno 0,8 con la original."""
    if drift is None:
        return E[k]
    return 0.8 * E[k] + 0.6 * E[drift]


def obs(
    tid: int,
    x: float,
    e: np.ndarray,
    team: Team = Team.A,
    number: int | None = None,
    libero: bool = False,
    y: float = 500.0,
) -> Observation:
    jersey = JerseyRead(number, 0.97) if number is not None else None
    return Observation(tid, (x, y, x + 60.0, y + 150.0), 0.9, team, e, jersey=jersey, libero=libero)


def field(
    numbers: list[int | None] | None = None, skip: set[int] = frozenset(), f: int = 0
) -> list[Observation]:
    """Los 6 jugadores de campo de A (tracks 1-6) en sus lugares; `skip` = índices ausentes."""
    nums = numbers or [None] * 6
    return [
        obs(k + 1, XS[k], emb(k), number=nums[k] if f % 2 == 0 else None) for k in range(6) if k not in skip
    ]


def by_track(out: FrameOutput) -> dict[str, str]:
    return {p.track_id: p.player_id for p in out.players}


def start(mgr: IdentityManager, frames: int = 30, numbers: list[int | None] | None = None) -> dict[str, str]:
    out = None
    for f in range(frames):
        out = mgr.update(f, field(numbers, f=f))
    assert out is not None
    ids = by_track(out)
    assert sorted(ids.values()) == [f"TEAM_A_PLAYER_0{k}" for k in range(1, 7)]
    return ids


def pids(mgr: IdentityManager) -> list[str]:
    return sorted(mgr.identity_states())


def test_ac1_returning_player_recovers_id_by_elimination() -> None:
    mgr = IdentityManager()
    ids = start(mgr)
    gone = ids["track_6"]
    for f in range(30, 230):  # ausente 200 frames (más que long_gap): la Re-ID estricta ya no aplica
        mgr.update(f, field(skip={5}, f=f))
    # vuelve lejos de donde se perdió y con otra apariencia
    outs = [
        mgr.update(f, [*field(skip={5}, f=f), obs(60, 300.0, emb(5, drift=20), y=800.0)])
        for f in range(230, 250)
    ]
    assert by_track(outs[-1])["track_60"] == gone
    assert "TEAM_A_PLAYER_08" not in pids(mgr)


def test_ac2_two_absent_players_recover_by_appearance() -> None:
    mgr = IdentityManager()
    ids = start(mgr)
    for f in range(30, 230):
        mgr.update(f, field(skip={4, 5}, f=f))
    # vuelven cambiados de lugar (el 5 donde estaba el 6 y al revés): decide la apariencia, no la posición
    outs = [
        mgr.update(
            f,
            [*field(skip={4, 5}, f=f), obs(50, XS[5], emb(4, drift=20)), obs(60, XS[4], emb(5, drift=21))],
        )
        for f in range(230, 250)
    ]
    got = by_track(outs[-1])
    assert got["track_50"] == ids["track_5"]
    assert got["track_60"] == ids["track_6"]


def test_ac3_libero_is_always_07_and_roles_never_mix() -> None:
    mgr = IdentityManager()
    out = None
    # 5 de campo + líbero (el 6.º de campo está afuera mientras juega el líbero)
    for f in range(60):
        out = mgr.update(f, [*field(skip={5}, f=f), obs(7, 900.0, emb(10), libero=True, y=300.0)])
    assert out is not None
    assert by_track(out)["track_7"] == "TEAM_A_PLAYER_07"
    # sale el líbero y entra el 6.º de campo: toma una identidad de campo, no la 07
    for f in range(60, 80):
        out = mgr.update(f, [*field(skip={5}, f=f), obs(16, XS[5], emb(5))])
    assert by_track(out)["track_16"] != "TEAM_A_PLAYER_07"
    # vuelve el líbero con otro track y otra luz (sale el 6.º): 07
    for f in range(80, 100):
        out = mgr.update(f, [*field(skip={5}, f=f), obs(17, 900.0, emb(10, drift=22), libero=True, y=300.0)])
    assert by_track(out)["track_17"] == "TEAM_A_PLAYER_07"
    assert "TEAM_A_PLAYER_08" not in pids(mgr)


def test_ac4_seventh_simultaneous_player_gets_no_identity() -> None:
    mgr = IdentityManager()
    start(mgr)
    out = None
    for f in range(30, 60):
        out = mgr.update(f, [*field(f=f), obs(70, 900.0, emb(11), y=300.0)])
    assert out is not None
    assert "track_70" not in by_track(out)


NUMS: list[int | None] = [1, 2, 3, 4, 5, 7]


def test_ac5_substitute_with_new_number_gets_08_and_outgoing_is_retired() -> None:
    mgr = IdentityManager()
    ids = start(mgr, numbers=NUMS)
    out_pid = ids["track_6"]
    for f in range(30, 60):
        mgr.update(f, field(NUMS, skip={5}, f=f))
    out = None
    for f in range(60, 90):  # entra el #14, que nadie tiene
        out = mgr.update(f, [*field(NUMS, skip={5}, f=f), obs(14, XS[5], emb(12), number=14)])
    assert out is not None
    assert by_track(out)["track_14"] == "TEAM_A_PLAYER_08"
    # vuelve alguien sin número legible: el 06 está retirado, no se le reasigna por descarte
    for f in range(90, 120):
        out = mgr.update(f, [*field(NUMS, skip={5}, f=f), obs(15, XS[5], emb(13))])
    assert by_track(out).get("track_15") != out_pid


def test_ac6_number_vetoes_elimination() -> None:
    mgr = IdentityManager()
    ids = start(mgr, numbers=NUMS)
    for f in range(30, 60):
        mgr.update(f, field(NUMS, skip={5}, f=f))
    out = None
    # un segundo tracklet con el #3 (el #3 está visible): no puede ser el #7 que falta
    for f in range(60, 90):
        out = mgr.update(f, [*field(NUMS, skip={5}, f=f), obs(30, XS[5], emb(14), number=3)])
    assert out is not None
    assert by_track(out).get("track_30") != ids["track_6"]
    assert "TEAM_A_PLAYER_08" not in pids(mgr)


def test_ac7_after_a_cut_everyone_recovers_their_id() -> None:
    mgr = IdentityManager()
    out = None
    b_xs = [200.0, 500.0, 800.0, 1100.0, 1400.0, 1700.0]
    for f in range(30):
        b = [obs(20 + k, b_xs[k], emb(6 + k), team=Team.B, y=150.0) for k in range(6)]
        out = mgr.update(f, [*field(f=f), *b])
    assert out is not None
    before = {p.player_id: p for p in out.players}
    mgr.reset()
    # otra toma: lugares mezclados, apariencia algo cambiada, tracks nuevos
    order = [3, 0, 5, 1, 4, 2]
    for f in range(30, 60):
        a = [obs(100 + k, XS[order[k]], emb(k, drift=24 + k)) for k in range(6)]
        b = [obs(200 + k, b_xs[order[k]], emb(6 + k, drift=24 + k), team=Team.B, y=150.0) for k in range(6)]
        out = mgr.update(f, [*a, *b])
    got = by_track(out)
    assert sorted(mgr.identity_states()) == sorted(before)  # ninguna identidad nueva
    first = {f"track_{k + 1}": k for k in range(6)}
    first.update({f"track_{20 + k}": 6 + k for k in range(6)})
    old = {k: by_track_pid for by_track_pid, k in ((p.player_id, first[p.track_id]) for p in before.values())}
    for k in range(6):
        assert got[f"track_{100 + k}"] == old[k]
        assert got[f"track_{200 + k}"] == old[6 + k]
