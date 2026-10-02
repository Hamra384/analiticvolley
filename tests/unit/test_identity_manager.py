"""Manejo de errores y casos borde del IdentityManager (SPEC-001, sección "Manejo de errores")."""

from __future__ import annotations

import numpy as np
import pytest

from volley_cv.identity import IdentityConfig, IdentityManager, JerseyRead, Observation, PlayerState, Team

EMB = np.eye(8, dtype=np.float32)[0]
OTHER = np.eye(8, dtype=np.float32)[1]


def obs(
    tid: int, x: float, emb: np.ndarray | None = EMB, team: Team | None = Team.A, **kw: object
) -> Observation:
    return Observation(
        track_id=tid, bbox=(x, 500.0, x + 60.0, 650.0), confidence=0.9, team=team, embedding=emb, **kw
    )


def feed(mgr: IdentityManager, frames: range, make: object) -> list[object]:
    return [mgr.update(f, make(f)) for f in frames]  # type: ignore[operator]


def test_frames_must_increase() -> None:
    mgr = IdentityManager()
    mgr.update(5, [])
    with pytest.raises(ValueError, match="posterior"):
        mgr.update(5, [])
    with pytest.raises(ValueError):
        mgr.update(3, [])


def test_invalid_bbox_is_dropped() -> None:
    mgr = IdentityManager()
    bad = Observation(track_id=1, bbox=(100, 100, 100, 200), confidence=0.9, team=Team.A, embedding=EMB)
    outs = feed(mgr, range(10), lambda f: [bad])
    assert all(o.players == [] for o in outs)  # type: ignore[attr-defined]


def test_zero_norm_embedding_is_treated_as_missing() -> None:
    mgr = IdentityManager()
    outs = feed(mgr, range(10), lambda f: [obs(1, 300 + f, emb=np.zeros(8, np.float32))])
    assert outs[-1].players[0].player_id == "TEAM_A_PLAYER_01"  # type: ignore[attr-defined]


def test_without_embeddings_motion_alone_reidentifies_nearby_return() -> None:
    mgr = IdentityManager()
    outs = feed(mgr, range(20), lambda f: [obs(1, 300 + 2 * f, emb=None)])
    outs += feed(mgr, range(20, 25), lambda f: [])
    outs += feed(mgr, range(25, 40), lambda f: [obs(2, 300 + 2 * f, emb=None)])
    ids = {p.player_id for o in outs for p in o.players}  # type: ignore[attr-defined]
    assert ids == {"TEAM_A_PLAYER_01"}
    assert outs[25].players[0].state == "REIDENTIFIED"  # type: ignore[attr-defined]


def test_far_return_without_embedding_is_not_forced_onto_old_identity() -> None:
    mgr = IdentityManager()
    feed(mgr, range(20), lambda f: [obs(1, 300, emb=None)])
    outs = feed(mgr, range(20, 40), lambda f: [obs(2, 1500, emb=None)])  # muy lejos, sin apariencia
    ids = {p.player_id for o in outs for p in o.players}  # type: ignore[attr-defined]
    assert "TEAM_A_PLAYER_01" not in ids


def test_reset_marks_identities_lost_and_reidentifies_by_appearance() -> None:
    mgr = IdentityManager()
    feed(mgr, range(20), lambda f: [obs(1, 300, emb=EMB)])
    mgr.reset()
    assert mgr.identity_states() == {"TEAM_A_PLAYER_01": PlayerState.LOST}
    # tras el corte reaparece en otra posición (otra toma) con la misma apariencia
    outs = feed(mgr, range(20, 30), lambda f: [obs(7, 1400, emb=EMB)])
    assert outs[0].players[0].player_id == "TEAM_A_PLAYER_01"  # type: ignore[attr-defined]
    assert outs[0].players[0].state == "REIDENTIFIED"  # type: ignore[attr-defined]


def test_long_gap_uses_appearance_only() -> None:
    cfg = IdentityConfig()
    mgr = IdentityManager(cfg)
    feed(mgr, range(10), lambda f: [obs(1, 300, emb=EMB)])
    gap_end = 10 + cfg.long_gap_frames + 5
    feed(mgr, range(10, gap_end), lambda f: [])
    outs = feed(mgr, range(gap_end, gap_end + 3), lambda f: [obs(9, 1500, emb=EMB)])
    assert outs[0].players[0].player_id == "TEAM_A_PLAYER_01"  # type: ignore[attr-defined]


def test_ambiguous_tracklet_waits_then_creates_identity() -> None:
    """Costo entre accept y new_identity: espera max_wait_frames y luego crea (hay cupo)."""
    cfg = IdentityConfig(max_wait_frames=10)
    mgr = IdentityManager(cfg)
    feed(mgr, range(10), lambda f: [obs(1, 300, emb=EMB)])
    feed(mgr, range(10, 30), lambda f: [])
    # coseno 0,45 con la identidad -> distancia 0,55 -> costo ~0,33: entre accept (0,30) y new_identity (0,40)
    k = float(np.sqrt(1 / 0.45**2 - 1))
    mixed = ((EMB + k * OTHER) / np.linalg.norm(EMB + k * OTHER)).astype(np.float32)
    outs = feed(mgr, range(30, 60), lambda f: [obs(2, 300, emb=mixed)])
    first = next(i for i, o in enumerate(outs) if o.players)  # type: ignore[attr-defined]
    created = outs[first].players[0]  # type: ignore[attr-defined]
    assert created.player_id == "TEAM_A_PLAYER_02"
    assert created.state == "DETECTED"
    # confirmado en el índice confirm_frames-1; espera max_wait_frames frames (incluido ese) y luego crea
    assert first == cfg.confirm_frames - 1 + cfg.max_wait_frames


def test_jersey_read_before_identity_is_kept() -> None:
    mgr = IdentityManager()
    outs = feed(mgr, range(8), lambda f: [obs(1, 300, jersey=JerseyRead(12, 0.95))])
    assert outs[-1].players[0].jersey_number == 12  # type: ignore[attr-defined]


def test_team_flip_on_same_track_splits_tracklet() -> None:
    mgr = IdentityManager()
    feed(mgr, range(10), lambda f: [obs(1, 300, emb=EMB, team=Team.A)])
    outs = feed(mgr, range(10, 30), lambda f: [obs(1, 300, emb=OTHER, team=Team.B)])
    last = outs[-1].players[0]  # type: ignore[attr-defined]
    assert last.team_id == "TEAM_B"
    assert last.track_id.startswith("track_1.")
