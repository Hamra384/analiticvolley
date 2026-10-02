"""Regresiones de la revisión independiente de S3 (hallazgos 1-10 del reviewer). SPEC-001."""

from __future__ import annotations

import numpy as np
import pytest

from volley_cv.identity import IdentityConfig, IdentityManager, JerseyRead, Observation, PlayerState, Team

from .synth import Scenario, SimPlayer, _iou, linear, reads, run, still
from .test_identity_scenarios import CFG, _overlap_windows, _waypoints


def _twelve(layout_seed: int, seed: int) -> Scenario:
    rng = np.random.default_rng(layout_seed)
    numbers = [3, 7, 9, 11, 14, 18]
    players = []
    for team, (y0, y1) in ((Team.A, (620, 980)), (Team.B, (180, 460))):
        for i in range(6):
            pts = [(rng.uniform(250, 1650), rng.uniform(y0, y1)) for _ in range(5)]
            jersey = reads(numbers[i], every=10) if team == Team.A else (lambda f: None)
            players.append(SimPlayer(f"{team}{i + 1}", team, _waypoints(pts, 600), jersey=jersey))
    return Scenario(frames=600, players=players, seed=seed, auto_swap_iou=0.3, auto_swap_prob=0.5)


# Hallazgo 1: AC-11 debe valer para cualquier semilla, no solo la elegida.
@pytest.mark.parametrize(("layout", "seed"), [(7, s) for s in range(10)] + [(9, s) for s in range(5)])
def test_finding1_ac11_holds_across_seeds(layout: int, seed: int) -> None:
    r = run(_twelve(layout, seed))
    assert len(r.all_ids()) == 12
    transient = total = 0
    majority = set()
    for gt, a in r.assignments().items():
        values = list(a.values())
        top = max(set(values), key=values.count)
        windows = _overlap_windows(r, gt)
        assert [f for f, pid in a.items() if pid != top and f not in windows] == [], gt
        transient += sum(1 for pid in values if pid != top)
        total += len(values)
        majority.add(top)
    assert len(majority) == 12
    assert transient / total < 0.05
    assert r.frames_with_duplicate_ids() == []


def test_finding1_three_player_chain_with_staggered_separation() -> None:
    """B5 se separa de B3 mientras B3 sigue superpuesto con B6 (y el tracker intercambia B3/B6)."""
    players = [
        SimPlayer("B3", Team.B, linear((800, 300), (900, 300), 0, 199)),
        SimPlayer("B5", Team.B, linear((830, 310), (500, 310), 0, 199)),
        SimPlayer("B6", Team.B, linear((1000, 290), (860, 290), 0, 199)),
    ]
    scn = Scenario(frames=200, players=players, swaps=[(5, "B3", "B5"), (40, "B3", "B6")])
    r = run(scn)
    for gt in ("B3", "B5", "B6"):
        a = r.assignments()[gt]
        windows = _overlap_windows(r, gt)
        tail = [pid for f, pid in a.items() if f > 150 and f not in windows]
        assert len(set(tail)) == 1, (gt, set(tail))
    finals = {r.player(gt, 199) for gt in ("B3", "B5", "B6")}
    assert len(finals) == 3


# Hallazgo 2: nunca se emite una caja con el equipo contrario al observado.
def test_finding2_box_never_reported_with_opposite_team() -> None:
    scn = Scenario(
        frames=140,
        players=[
            SimPlayer("P1", Team.A, linear((400, 700), (1000, 700), 0, 139)),
            SimPlayer("P2", Team.B, linear((1000, 715), (400, 715), 0, 139)),
        ],
        swaps=[(67, "P1", "P2")],  # intercambio al empezar la superposición
    )
    r = run(scn)
    for gt, team in (("P1", "TEAM_A"), ("P2", "TEAM_B")):
        for f, pid in r.assignments()[gt].items():
            assert pid.startswith(team), (gt, f, pid)


# Hallazgo 3: track_id duplicado en un frame no duplica identidades.
def test_finding3_duplicate_track_id_in_frame() -> None:
    mgr = IdentityManager()
    emb = np.eye(8, dtype=np.float32)[0]
    for f in range(10):
        a = Observation(1, (100.0, 500.0, 160.0, 650.0), 0.9, Team.A, emb)
        b = Observation(1, (900.0, 500.0, 960.0, 650.0), 0.5, Team.A, emb)
        out = mgr.update(f, [a, b])
        ids = [p.player_id for p in out.players]
        assert len(ids) == len(set(ids)), (f, ids)
    assert len(mgr.identity_states()) == 1


# Hallazgo 4: una detección espuria de 1 frame sin apariencia no roba una identidad oculta.
def test_finding4_single_spurious_detection_does_not_steal_occluded_identity() -> None:
    mgr = IdentityManager()
    emb = np.eye(8, dtype=np.float32)[0]
    for f in range(20):
        mgr.update(f, [Observation(1, (300.0 + f, 500.0, 360.0 + f, 650.0), 0.9, Team.A, emb)])
    mgr.update(20, [])
    out = mgr.update(21, [Observation(99, (330.0, 500.0, 390.0, 650.0), 0.6, Team.A, None)])
    assert out.players == []
    out = mgr.update(22, [Observation(1, (322.0, 500.0, 382.0, 650.0), 0.9, Team.A, emb)])
    assert out.players[0].player_id == "TEAM_A_PLAYER_01"
    assert out.players[0].track_id == "track_1"


# Hallazgo 5: las lecturas de dorsal no se cuentan dos veces al re-vincular.
def test_finding5_jersey_votes_not_double_counted_when_original_tracklet_returns() -> None:
    """Reproducción del reviewer: tracklet 1 lee 10 veces, otro tracklet toma la identidad, el 1 vuelve."""
    e = np.eye(8, dtype=np.float32)[0]

    def o(t: int, x: float, **kw: object) -> Observation:
        return Observation(t, (x, 500.0, x + 60.0, 650.0), 0.9, Team.A, e, **kw)  # type: ignore[arg-type]

    mgr = IdentityManager()
    for f in range(10):
        mgr.update(f, [o(1, 300, jersey=JerseyRead(4, 0.9))])
    mgr.update(12, [o(2, 305)])
    for f in range(14, 16):
        mgr.update(f, [o(1, 310)])
    idn = mgr._identities["TEAM_A_PLAYER_01"]
    assert idn.votes.counts[4] == 10


def test_finding5_jersey_votes_not_double_counted_on_relink() -> None:
    p = SimPlayer(
        "A1", Team.A, linear((300, 700), (700, 700), 0, 99), absent=set(range(40, 70)), jersey=reads(4)
    )
    mgr = IdentityManager()
    run(Scenario(frames=100, players=[p]), manager=mgr)
    idn = next(iter(mgr._identities.values()))
    present_reads = sum(1 for f in range(100) if f % 5 == 0 and not 40 <= f < 70)
    assert idn.votes.counts[4] == present_reads


# Hallazgo 6: tras reset() las identidades siguen LOST y no bloquean el cupo.
def test_finding6_reset_keeps_lost_and_frees_roster() -> None:
    mgr = IdentityManager()
    emb = np.eye(16, dtype=np.float32)
    for f in range(10):
        mgr.update(
            f,
            [
                Observation(i, (100.0 + 200 * i, 600.0, 160.0 + 200 * i, 750.0), 0.9, Team.A, emb[i])
                for i in range(6)
            ],
        )
    mgr.reset()
    new = Observation(50, (800.0, 300.0, 860.0, 450.0), 0.9, Team.A, emb[10])
    outs = [mgr.update(f, [new]) for f in range(10, 10 + CFG.confirm_frames)]
    assert all(
        s == PlayerState.LOST
        for pid, s in mgr.identity_states().items()
        if pid != outs[-1].players[0].player_id
    )
    assert outs[-1].players[0].state == "DETECTED"


# Hallazgo 9: cambio de track sin hueco no se reporta como REIDENTIFIED.
def test_finding9_track_id_change_without_gap_stays_tracked() -> None:
    mgr = IdentityManager()
    emb = np.eye(8, dtype=np.float32)[0]
    for f in range(20):
        mgr.update(f, [Observation(1, (300.0 + f, 500.0, 360.0 + f, 650.0), 0.9, Team.A, emb)])
    out = mgr.update(20, [Observation(2, (320.0, 500.0, 380.0, 650.0), 0.9, Team.A, emb)])
    assert out.players[0].player_id == "TEAM_A_PLAYER_01"
    assert out.players[0].state == "TRACKED"


# Hallazgo 10: tests más estrictos.
def test_finding10_ac7_every_gap_frame_is_occluded() -> None:
    mgr = IdentityManager()
    emb = np.eye(8, dtype=np.float32)[0]
    for f in range(50):
        mgr.update(f, [Observation(1, (300.0 + f, 500.0, 360.0 + f, 650.0), 0.9, Team.A, emb)])
    for f in range(50, 55):
        mgr.update(f, [])
        assert mgr.identity_states() == {"TEAM_A_PLAYER_01": PlayerState.OCCLUDED}, f


def test_finding10_unknown_team_then_known_creates_identity() -> None:
    mgr = IdentityManager()
    emb = np.eye(8, dtype=np.float32)[0]
    for f in range(10):
        assert mgr.update(f, [Observation(1, (300.0, 500.0, 360.0, 650.0), 0.9, None, emb)]).players == []
    outs = [
        mgr.update(f, [Observation(1, (300.0, 500.0, 360.0, 650.0), 0.9, Team.B, emb)]) for f in range(10, 15)
    ]
    assert outs[-1].players[0].player_id == "TEAM_B_PLAYER_01"


def test_finding10_roster_counts_occluded_identities() -> None:
    players = [SimPlayer(f"A{i}", Team.A, still((250 + 220 * i, 750))) for i in range(5)]
    players.append(SimPlayer("A5", Team.A, still((1350, 750)), absent=set(range(60, 70))))
    players.append(SimPlayer("A7", Team.A, still((1700, 950), 61, 200)))  # aparece mientras A5 está OCCLUDED
    r = run(Scenario(frames=200, players=players))
    assert len({pid for pid in r.all_ids() if pid.startswith("TEAM_A")}) == 6
    assert r.ids("A5") and len(r.ids("A5")) == 1


def test_finding10_swap_at_overlap_start_is_resolved() -> None:
    scn = Scenario(
        frames=140,
        players=[
            SimPlayer("P1", Team.A, linear((400, 700), (1000, 700), 0, 139)),
            SimPlayer("P2", Team.A, linear((1000, 715), (400, 715), 0, 139)),
        ],
        swaps=[(67, "P1", "P2")],
    )
    r = run(scn)
    first_overlap = next(f for f in range(140) if _iou(*list(r.truth[f])[:2]) > CFG.overlap_iou)
    assert 66 <= first_overlap <= 68
    before = {gt: r.player(gt, 30) for gt in ("P1", "P2")}
    for f in range(100, 140):
        assert r.player("P1", f) == before["P1"] and r.player("P2", f) == before["P2"]


def test_finding7_stale_partners_do_not_reassign_far_apart_players() -> None:
    """Dos jugadores que se cruzaron hace mucho no se reasignan al volver a estar libres."""
    p1 = SimPlayer(
        "P1", Team.A, _waypoints([(400, 700), (1000, 700), (1600, 700)], 400), absent=set(range(150, 300))
    )
    p2 = SimPlayer("P2", Team.A, linear((1000, 715), (300, 715), 0, 399))
    r = run(Scenario(frames=400, players=[p1, p2]))
    assert len(r.ids("P1")) == 1 and len(r.ids("P2")) == 1


def test_config_validates_new_reid_minimum() -> None:
    assert IdentityConfig().reid_min_obs_without_embedding >= 2
    _ = JerseyRead(1, 0.9)
