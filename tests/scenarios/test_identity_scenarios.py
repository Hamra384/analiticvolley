"""Escenarios de aceptación del IdentityManager (SPEC-001, docs/specs/001-identity-manager.md).

Cada test cita el AC que cubre. Los escenarios son deterministas (semilla fija) y no requieren video ni GPU.
"""

from __future__ import annotations

import numpy as np

from volley_cv.identity import IdentityConfig, IdentityManager, JerseyRead, PlayerState, Team

from .synth import Scenario, SimPlayer, linear, reads, run, still

CFG = IdentityConfig()


def test_ac1_normal_tracking_single_identity() -> None:
    r = run(Scenario(frames=300, players=[SimPlayer("A1", Team.A, linear((300, 700), (900, 700), 0, 299))]))
    a = r.assignments()["A1"]
    assert min(a) == CFG.confirm_frames - 1
    assert len(set(a.values())) == 1
    assert r.find("A1", CFG.confirm_frames - 1).state == PlayerState.DETECTED  # type: ignore[union-attr]
    for f in range(CFG.confirm_frames, 300):
        assert r.find("A1", f).state == PlayerState.TRACKED  # type: ignore[union-attr]
    assert next(iter(a.values())).startswith("TEAM_A_PLAYER_")


def test_ac2_occlusion_keeps_identity_and_reports_reidentified() -> None:
    p = SimPlayer("A1", Team.A, linear((300, 700), (900, 700), 0, 199), absent=set(range(80, 110)))
    r = run(Scenario(frames=200, players=[p]))
    assert len(r.ids("A1")) == 1
    assert r.find("A1", 110).state == PlayerState.REIDENTIFIED  # type: ignore[union-attr]
    assert r.find("A1", 111).state == PlayerState.TRACKED  # type: ignore[union-attr]
    assert all(r.player("A1", f) is None for f in range(80, 110))


def _crossing(team2: Team, swap_frame: int = 60) -> Scenario:
    return Scenario(
        frames=140,
        players=[
            SimPlayer("P1", Team.A, linear((400, 700), (1000, 700), 0, 139)),
            SimPlayer("P2", team2, linear((1000, 715), (400, 715), 0, 139)),
        ],
        swaps=[(swap_frame, "P1", "P2")],
    )


def _separation_frame(r: object, frames: int = 140) -> int:
    # primer frame después del cruce en que las cajas ya no se superponen (IoU <= overlap_iou)
    from .synth import _iou

    truth = r.truth  # type: ignore[attr-defined]
    seen_overlap = False
    for f in range(frames):
        boxes = list(truth[f])
        if len(boxes) == 2:
            iou = _iou(boxes[0], boxes[1])
            if iou > CFG.overlap_iou:
                seen_overlap = True
            elif seen_overlap:
                return f
    raise AssertionError("no hubo separación")


def test_ac3_same_team_crossing_with_tracker_swap_keeps_identities() -> None:
    r = run(_crossing(Team.A))
    before = {gt: r.player(gt, 30) for gt in ("P1", "P2")}
    assert before["P1"] != before["P2"]
    settle = _separation_frame(r) + CFG.separation_frames
    for f in range(settle, 140):
        assert r.player("P1", f) == before["P1"], f"P1 cambió de identidad en frame {f}"
        assert r.player("P2", f) == before["P2"], f"P2 cambió de identidad en frame {f}"
    assert r.frames_with_duplicate_ids() == []
    assert len(r.all_ids()) == 2


def test_ac17_cross_team_crossing_never_changes_team() -> None:
    r = run(_crossing(Team.B))
    settle = _separation_frame(r) + CFG.separation_frames
    for gt, team in (("P1", "TEAM_A"), ("P2", "TEAM_B")):
        for f, pid in r.assignments()[gt].items():
            if f >= settle:
                assert pid.startswith(team), (gt, f, pid)
    assert r.frames_with_duplicate_ids() == []
    assert len(r.all_ids()) == 2


def test_ac4_visible_number_is_assigned() -> None:
    p = SimPlayer("A1", Team.A, still((600, 700)), jersey=reads(7, conf=0.95, every=3))
    r = run(Scenario(frames=60, players=[p]))
    last = r.find("A1", 59)
    assert last.jersey_number == 7  # type: ignore[union-attr]
    assert last.jersey_confidence >= 0.9  # type: ignore[union-attr]


def test_ac5_invisible_number_stays_null_with_stable_id() -> None:
    r = run(Scenario(frames=150, players=[SimPlayer("B1", Team.B, linear((600, 300), (900, 320), 0, 149))]))
    assert len(r.ids("B1")) == 1
    for f in r.assignments()["B1"]:
        p = r.find("B1", f)
        assert p.jersey_number is None and p.jersey_confidence == 0.0  # type: ignore[union-attr]
    assert r.ids("B1")[0].startswith("TEAM_B_PLAYER_")


def test_ac6_number_appearing_later_keeps_identity() -> None:
    p = SimPlayer("B3", Team.B, linear((500, 300), (1000, 300), 0, 299), jersey=reads(8, start=200, every=4))
    r = run(Scenario(frames=300, players=[p]))
    assert len(r.ids("B3")) == 1
    assert r.find("B3", 150).jersey_number is None  # type: ignore[union-attr]
    assert r.find("B3", 299).jersey_number == 8  # type: ignore[union-attr]


def test_ac7_short_detector_dropout_keeps_track_and_identity() -> None:
    p = SimPlayer("A1", Team.A, linear((300, 700), (700, 700), 0, 99), absent=set(range(50, 55)))
    r = run(Scenario(frames=100, players=[p]))
    assert len(r.ids("A1")) == 1
    assert r.find("A1", 55).state == PlayerState.TRACKED  # type: ignore[union-attr]
    # durante el hueco la identidad sigue existiendo (OCCLUDED), sin caja en la salida
    mgr_state = _states_during_gap(p, range(50, 55))
    assert set(mgr_state) == {PlayerState.OCCLUDED}


def _states_during_gap(p: SimPlayer, gap: range) -> list[PlayerState]:
    mgr = IdentityManager()
    run(Scenario(frames=gap.stop, players=[p]), manager=mgr)
    states = mgr.identity_states()
    return [states[pid] for pid in states]


def test_ac8_spurious_two_frame_tracklet_creates_no_identity() -> None:
    r = run(Scenario(frames=50, players=[SimPlayer("X", Team.A, still((600, 700), 10, 11))]))
    assert r.all_ids() == set()


def test_ac9_new_player_with_room_gets_new_identity() -> None:
    players = [
        SimPlayer("A1", Team.A, linear((300, 700), (500, 700), 0, 199)),
        SimPlayer("A2", Team.A, linear((1200, 800), (1000, 800), 100, 199)),
    ]
    r = run(Scenario(frames=200, players=players))
    assert len(r.ids("A2")) == 1
    assert r.ids("A2")[0] != r.ids("A1")[0]


def test_ac10_substitution_does_not_reuse_outgoing_identity() -> None:
    players = [
        SimPlayer("OUT", Team.A, linear((900, 700), (1800, 700), 0, 99)),
        SimPlayer("IN", Team.A, linear((1800, 700), (900, 700), 140, 260)),
    ]
    r = run(Scenario(frames=261, players=players))
    assert len(r.ids("IN")) == 1
    assert r.ids("IN")[0] != r.ids("OUT")[0]


def test_ac11_team_without_numbers_keeps_12_stable_ids() -> None:
    rng = np.random.default_rng(7)
    numbers = [3, 7, 9, 11, 14, 18]
    players = []
    for team, (y0, y1) in ((Team.A, (620, 980)), (Team.B, (180, 460))):
        for i in range(6):
            pts = [(rng.uniform(250, 1650), rng.uniform(y0, y1)) for _ in range(5)]
            path = _waypoints(pts, 600)
            jersey = (
                reads(numbers[i], every=10)
                if team == Team.A
                else (reads(9, start=450, every=6) if i == 2 else (lambda f: None))
            )
            players.append(SimPlayer(f"{team}{i + 1}", team, path, jersey=jersey))
    r = run(Scenario(frames=600, players=players, seed=3, auto_swap_iou=0.3, auto_swap_prob=0.5))

    assert len(r.all_ids()) == 12
    # Igual que AC-3: durante una superposición la salida puede reflejar el intercambio del tracker;
    # se corrige a más tardar `separation_frames` frames tras separarse. Fuera de esas ventanas: 0 errores.
    majority = {}
    transient = total = 0
    for gt, a in r.assignments().items():
        values = list(a.values())
        top = max(set(values), key=values.count)
        windows = _overlap_windows(r, gt)
        wrong_outside = [f for f, pid in a.items() if pid != top and f not in windows]
        assert wrong_outside == [], (gt, wrong_outside)
        transient += sum(1 for f, pid in a.items() if pid != top)
        total += len(a)
        majority[gt] = top
    assert len(set(majority.values())) == 12
    assert transient / total < 0.05, f"errores transitorios en superposiciones: {transient / total:.1%}"
    assert r.frames_with_duplicate_ids() == []
    for i in range(6):
        p = r.find(f"B{i + 1}", 599)
        expected = 9 if i == 2 else None
        assert p.jersey_number == expected, (i, p.jersey_number)  # type: ignore[union-attr]
    for i, n in enumerate(numbers):
        assert r.find(f"A{i + 1}", 599).jersey_number == n  # type: ignore[union-attr]


def _overlap_windows(r: object, gt: str) -> set[int]:
    """Frames en que `gt` está superpuesto (IoU > overlap_iou) o hasta `separation_frames` tras separarse."""
    from .synth import _iou

    truth = r.truth  # type: ignore[attr-defined]
    frames: set[int] = set()
    last_overlap = None
    for f in sorted(truth):
        boxes = {g: b for b, g in truth[f].items()}
        if gt not in boxes:
            continue
        mine = boxes[gt]
        if any(g != gt and _iou(mine, b) > CFG.overlap_iou for g, b in boxes.items()):
            last_overlap = f
            frames.add(f)
        elif last_overlap is not None and f - last_overlap <= CFG.separation_frames:
            frames.add(f)
    return frames


def _waypoints(pts: list[tuple[float, float]], frames: int):  # type: ignore[no-untyped-def]
    seg = frames / (len(pts) - 1)

    def at(f: int) -> tuple[float, float] | None:
        if f < 0 or f >= frames:
            return None
        k = min(int(f // seg), len(pts) - 2)
        t = (f - k * seg) / seg
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        return x0 + t * (x1 - x0), y0 + t * (y1 - y0)

    return at


def test_ac12_single_low_confidence_read_is_ignored() -> None:
    p = SimPlayer("A1", Team.A, still((600, 700)), jersey=lambda f: JerseyRead(1, 0.6) if f == 20 else None)
    r = run(Scenario(frames=60, players=[p]))
    assert r.find("A1", 59).jersey_number is None  # type: ignore[union-attr]


def test_ac13_jersey_unique_per_team_but_shared_across_teams() -> None:
    players = [
        SimPlayer("A1", Team.A, still((400, 700)), jersey=reads(5, every=3)),
        SimPlayer("A2", Team.A, still((1200, 700)), jersey=reads(5, every=6)),
        SimPlayer("B1", Team.B, still((800, 300)), jersey=reads(5, every=3)),
    ]
    r = run(Scenario(frames=90, players=players))
    assert r.find("A1", 89).jersey_number == 5  # type: ignore[union-attr]
    assert r.find("A2", 89).jersey_number is None  # type: ignore[union-attr]
    assert r.find("B1", 89).jersey_number == 5  # type: ignore[union-attr]


def test_ac14_full_roster_does_not_create_seventh_identity() -> None:
    players = [SimPlayer(f"A{i}", Team.A, still((250 + 220 * i, 750))) for i in range(6)]
    players.append(SimPlayer("A7", Team.A, still((1700, 900), 60, 200)))
    r = run(Scenario(frames=200, players=players))
    assert len({pid for pid in r.all_ids() if pid.startswith("TEAM_A")}) == 6
    assert r.assignments().get("A7", {}) == {}


def test_ac16_deterministic() -> None:
    def scn() -> Scenario:
        return Scenario(
            frames=150,
            players=[
                SimPlayer("P1", Team.A, linear((400, 700), (1000, 700), 0, 149)),
                SimPlayer("P2", Team.A, linear((1000, 715), (400, 715), 0, 149), absent=set(range(90, 100))),
            ],
            swaps=[(70, "P1", "P2")],
        )

    a, b = run(scn()), run(scn())
    assert [o.model_dump() for o in a.outputs] == [o.model_dump() for o in b.outputs]


def test_unknown_team_tracklet_waits() -> None:
    p = SimPlayer("U", Team.A, still((600, 700)), team_visible=False)
    r = run(Scenario(frames=40, players=[p]))
    assert r.all_ids() == set()
