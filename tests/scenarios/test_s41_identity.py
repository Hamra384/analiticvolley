"""S4.1 (#11): criterios AC-18…AC-22 de SPEC-001, a partir de los problemas vistos en el debug real de A2."""

from __future__ import annotations

import numpy as np

from volley_cv.identity import IdentityConfig, IdentityManager, Observation, PlayerState, Team

E = np.eye(16, dtype=np.float32)


def obs(
    tid: int, x: float, e: int, team: Team | None = Team.A, in_court: bool = True, y: float = 650.0
) -> Observation:
    return Observation(tid, (x, y - 150, x + 60, y), 0.9, team, E[e], in_court=in_court)


def test_ac18_never_on_court_creates_no_identity() -> None:
    """Un juez de línea parado en la zona libre (nunca pisa la cancha) no recibe identidad."""
    mgr = IdentityManager()
    outs = [mgr.update(f, [obs(1, 1700, 0, in_court=False)]) for f in range(60)]
    assert all(o.players == [] for o in outs)


def test_ac18_existing_identity_is_followed_off_court() -> None:
    """Un jugador que sale a sacar (fuera de la cancha) conserva su identidad."""
    mgr = IdentityManager()
    for f in range(20):
        mgr.update(f, [obs(1, 800 + f, 0)])
    outs = [mgr.update(f, [obs(1, 820 + f, 0, in_court=False)]) for f in range(20, 80)]
    assert {p.player_id for o in outs for p in o.players} == {"TEAM_A_PLAYER_01"}
    assert all(len(o.players) == 1 for o in outs)


def test_ac19_mixed_team_votes_do_not_create_identity_until_consistent() -> None:
    """Votos A/B mezclados (como el japonés blanco que nació como Argentina en A2) no crean identidad."""
    cfg = IdentityConfig()
    mgr = IdentityManager(cfg)
    teams = [Team.B, Team.A, Team.A, Team.B, Team.A, Team.A, Team.B, Team.A]  # 62 % A
    for f, t in enumerate(teams):
        assert mgr.update(f, [obs(1, 800, 0, team=t)]).players == []
    out = None
    for f in range(len(teams), len(teams) + 30):  # luego votos consistentes de A
        out = mgr.update(f, [obs(1, 800, 0, team=Team.A)])
        if out.players:
            break
    assert out is not None and out.players and out.players[0].team_id == "TEAM_A"


def test_ac20_intermittent_contradictions_with_unknown_frames_split_tracklet() -> None:
    mgr = IdentityManager()
    for f in range(20):
        mgr.update(f, [obs(1, 800, 0, team=Team.B)])
    # el tracker pasa a seguir a un jugador de A, con frames sin equipo intercalados
    # misma apariencia a propósito: aísla la lógica de equipo (sin partición por apariencia)
    pattern = [Team.A, None, Team.A, None, Team.A, None, Team.A]
    outs = [mgr.update(20 + k, [obs(1, 800, 0, team=t)]) for k, t in enumerate(pattern * 6)]
    teams_out = {p.team_id for o in outs[-10:] for p in o.players}
    assert "TEAM_B" not in teams_out  # la caja ya no se reporta como B


def test_ac21_full_roster_with_occluded_identity_admits_real_player() -> None:
    cfg = IdentityConfig(max_wait_frames=10)
    mgr = IdentityManager(cfg)
    for f in range(20):
        mgr.update(f, [obs(i, 100 + 250 * i, i) for i in range(6)])
    # la identidad 5 (p. ej. un falso positivo) desaparece; aparece un jugador real nuevo en cancha
    out = None
    for f in range(20, 70):
        out = mgr.update(f, [*(obs(i, 100 + 250 * i, i) for i in range(5)), obs(50, 900, 10, y=400)])
    assert out is not None
    new = [p for p in out.players if p.track_id == "track_50"]
    assert new and new[0].player_id == "TEAM_A_PLAYER_07"
    assert mgr.identity_states()["TEAM_A_PLAYER_06"] == PlayerState.LOST


def test_ac22_long_gap_requires_strict_appearance_match() -> None:
    """Tras un hueco largo, alguien solo 'parecido' (juez de línea) no hereda la identidad."""
    cfg = IdentityConfig()
    mgr = IdentityManager(cfg)
    for f in range(20):
        mgr.update(f, [obs(1, 800, 0)])
    gap_end = 20 + cfg.long_gap_frames + 10
    for f in range(20, gap_end):
        mgr.update(f, [])
    # coseno 0,9 con el jugador: costo 0,6*0,1 + 0,4*0,5 = 0,26 pasaría el umbral normal (0,30), pero la
    # distancia de apariencia (0,1) supera la estricta para huecos largos (appearance_only_max_dist = 0,08)
    k = float(np.sqrt(1 / 0.9**2 - 1))
    similar = (E[0] + k * E[5]) / np.linalg.norm(E[0] + k * E[5])
    outs = [
        mgr.update(
            f, [Observation(9, (1500.0, 500.0, 1560.0, 650.0), 0.9, Team.A, similar.astype(np.float32))]
        )
        for f in range(gap_end, gap_end + 10)
    ]
    assert all(p.player_id != "TEAM_A_PLAYER_01" for o in outs for p in o.players)


def test_h2_split_fragment_does_not_inherit_in_court_evidence() -> None:
    """Revisión S4.1 H2: el tracker salta (mismo id) de un jugador en cancha a un juez siempre afuera."""
    mgr = IdentityManager()
    for f in range(20):
        mgr.update(f, [obs(1, 800, 0)])
    outs = [mgr.update(f, [obs(1, 1700, 7, in_court=False)]) for f in range(20, 80)]
    assert all(p.player_id == "TEAM_A_PLAYER_01" or False for o in outs[:5] for p in o.players)
    assert "TEAM_A_PLAYER_02" not in {p.player_id for o in outs for p in o.players}


def test_m1_off_court_lookalike_cannot_take_recently_lost_identity_by_motion() -> None:
    """Revisión S4.1 M1: alguien en la zona libre, cerca y parecido, no hereda una identidad recién perdida"""
    mgr = IdentityManager()
    for f in range(20):
        mgr.update(f, [obs(1, 800, 0)])
    for f in range(20, 40):
        mgr.update(f, [])
    k = float(np.sqrt(1 / 0.85**2 - 1))  # distancia de apariencia 0,15 (parecido realista según SPIKE-003)
    look = ((E[0] + k * E[6]) / np.linalg.norm(E[0] + k * E[6])).astype(np.float32)
    outs = [
        mgr.update(f, [Observation(5, (1000.0, 500.0, 1060.0, 650.0), 0.9, Team.A, look, in_court=False)])
        for f in range(40, 50)
    ]
    assert "TEAM_A_PLAYER_01" not in {p.player_id for o in outs for p in o.players}


def test_ac22_long_gap_off_court_person_cannot_inherit_identity() -> None:
    """Revisión visual S4.1 (A2): alguien del staff, afuera, "heredó" la identidad de un jugador perdido."""
    cfg = IdentityConfig()
    mgr = IdentityManager(cfg)
    for f in range(20):
        mgr.update(f, [obs(1, 800, 0)])
    gap_end = 20 + cfg.long_gap_frames + 10
    for f in range(20, gap_end):
        mgr.update(f, [])
    outs = [mgr.update(f, [obs(9, 1700, 0, in_court=False)]) for f in range(gap_end, gap_end + 10)]
    assert all(p.player_id != "TEAM_A_PLAYER_01" for o in outs for p in o.players)


def test_ac22_long_gap_identical_appearance_still_reidentifies() -> None:
    cfg = IdentityConfig()
    mgr = IdentityManager(cfg)
    for f in range(20):
        mgr.update(f, [obs(1, 800, 0)])
    gap_end = 20 + cfg.long_gap_frames + 10
    for f in range(20, gap_end):
        mgr.update(f, [])
    out = mgr.update(gap_end, [obs(9, 1500, 0)])
    assert out.players and out.players[0].player_id == "TEAM_A_PLAYER_01"
