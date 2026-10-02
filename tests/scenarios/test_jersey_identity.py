"""SPEC-003 AC-1, AC-2, AC-3, AC-5: número de camiseta como señal de identidad."""

from __future__ import annotations

import numpy as np

from volley_cv.identity import IdentityConfig, IdentityManager, JerseyRead, Observation, PlayerState, Team

E = np.eye(16, dtype=np.float32)
CFG = IdentityConfig()


def obs(tid: int, x: float, e: int, number: int | None = None, team: Team = Team.A) -> Observation:
    jersey = JerseyRead(number, 0.97) if number is not None else None
    return Observation(tid, (x, 500.0, x + 60.0, 650.0), 0.9, team, E[e], jersey=jersey)


def test_ac1_provisional_identity_merges_into_original_when_number_confirms() -> None:
    mgr = IdentityManager()
    for f in range(30):
        mgr.update(f, [obs(1, 800, 0, number=7 if f % 2 == 0 else None)])
    # desaparece; vuelve mucho después con otra apariencia (luz, pose): la Re-ID por apariencia no alcanza
    gap_end = 30 + CFG.long_gap_frames + 10
    for f in range(30, gap_end):
        mgr.update(f, [])
    # vuelve de frente (sin número visible) y recién después muestra la espalda con el 7
    outs = [
        mgr.update(f, [obs(2, 300, 3, number=7 if f >= gap_end + 15 and f % 2 == 0 else None)])
        for f in range(gap_end, gap_end + 40)
    ]
    provisional = outs[CFG.confirm_frames].players[0].player_id
    assert provisional != "TEAM_A_PLAYER_01"  # al principio es una identidad provisoria
    assert outs[-1].players[0].player_id == "TEAM_A_PLAYER_01"  # el número la fusiona en la original
    assert [(src, dst) for src, dst, _ in mgr.merges] == [(provisional, "TEAM_A_PLAYER_01")]
    assert provisional not in mgr.identity_states()


def test_ac2_coexisting_identities_with_same_number_do_not_merge() -> None:
    mgr = IdentityManager()
    outs = [
        mgr.update(
            f,
            [
                obs(1, 300, 0, number=7 if f % 2 == 0 else None),
                obs(2, 1200, 1, number=7 if f % 3 == 0 else None),
            ],
        )
        for f in range(60)
    ]
    assert mgr.merges == []
    last = {p.track_id: p for p in outs[-1].players}
    assert len({p.player_id for p in last.values()}) == 2
    assert sorted(p.jersey_number for p in last.values() if p.jersey_number is not None) == [7]  # RF-7


def test_ac3_same_number_different_teams_never_merge() -> None:
    mgr = IdentityManager()
    for f in range(30):
        mgr.update(f, [obs(1, 800, 0, number=7 if f % 2 == 0 else None, team=Team.A)])
    for f in range(30, 40):
        mgr.update(f, [])
    outs = [
        mgr.update(f, [obs(2, 300, 3, number=7 if f % 2 == 0 else None, team=Team.B)]) for f in range(40, 80)
    ]
    assert mgr.merges == []
    assert outs[-1].players[0].player_id.startswith("TEAM_B")


def test_ac5_number_change_on_same_track_splits_tracklet() -> None:
    mgr = IdentityManager()
    for f in range(20):
        mgr.update(f, [obs(1, 800, 0, number=7 if f % 2 == 0 else None)])
    # el tracker pasa a otro jugador con la misma apariencia de color (compañero), que lleva el 11
    outs = [mgr.update(f, [obs(1, 800, 0, number=11)]) for f in range(20, 40)]
    assert outs[-1].players[0].track_id.startswith("track_1.")
    first = mgr._identities["TEAM_A_PLAYER_01"]
    assert first.votes.counts.get(11, 0) == 0  # el 11 no se sumó a la identidad del 7
    assert first.jersey is not None and first.jersey[0] == 7


def test_ac4_merge_carries_orphan_votes_and_overlap_partners() -> None:
    mgr = IdentityManager()
    for f in range(20):
        mgr.update(f, [obs(1, 800, 0)])
    back = 20 + CFG.long_gap_frames + 10
    for f in range(20, back):
        mgr.update(f, [])
    for f in range(back, back + 20):
        mgr.update(f, [obs(2, 300, 3), obs(3, 1200, 5, team=Team.B)])
    src, dst = "TEAM_A_PLAYER_02", "TEAM_A_PLAYER_01"
    assert {src, dst} <= set(mgr.identity_states())
    # votos que quedaron en la identidad aunque su tracklet ya no exista, y una superposición abierta con B
    mgr._identities[src].votes.add(JerseyRead(7, 0.97))
    mgr._partners[src] = {"TEAM_B_PLAYER_01"}
    mgr._partners["TEAM_B_PLAYER_01"] = {src}
    mgr._merge(src, dst, back + 20)
    assert src not in mgr.identity_states()
    assert mgr._identities[dst].votes.counts[7] == 1
    assert mgr._partners["TEAM_B_PLAYER_01"] == {dst}
    assert mgr._partners[dst] == {"TEAM_B_PLAYER_01"}  # simétrico (revisión S5a B1)
    assert src not in mgr._partners


def _merged_after_gap(mgr: IdentityManager, read_every: int) -> tuple[int, list[str]]:
    """P01 (track 1, #7) desaparece; vuelve como track 2 con otra apariencia y se fusiona por el 7."""
    for f in range(30):
        mgr.update(f, [obs(1, 800, 0, number=7 if f % 2 == 0 else None)])
    back = 30 + CFG.long_gap_frames + 10
    for f in range(30, back):
        mgr.update(f, [])
    ids = []
    for f in range(back, back + 30):
        out = mgr.update(f, [obs(2, 300, 3, number=7 if f % read_every == 0 else None)])
        ids += [p.player_id for p in out.players]
    assert mgr.merges, "el escenario debe producir la fusión"
    return back + 30, ids


def test_review_h1_merge_leaves_a_single_tracklet_linked_no_duplicate_ids() -> None:
    mgr = IdentityManager()
    f, _ = _merged_after_gap(mgr, read_every=2)
    # el tracker revive el ID viejo (track 1) junto al 2 en el mismo frame
    out = mgr.update(f, [obs(2, 300, 3), obs(1, 1200, 0)])
    ids = [p.player_id for p in out.players]
    assert len(ids) == len(set(ids))
    assert sum(t.player == "TEAM_A_PLAYER_01" for t in mgr._tracklets.values()) == 1


def test_review_m1_merge_at_creation_is_reidentified_then_tracked() -> None:
    mgr = IdentityManager()
    _merged_after_gap(mgr, read_every=1)  # lecturas en cada frame: se fusiona en el frame de creación
    states = [s for s in mgr.identity_states().values()]
    assert PlayerState.DETECTED not in states
