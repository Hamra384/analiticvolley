"""IdentityManager: identidad persistente de jugadores sobre tracklets de corto plazo (SPEC-001).

Flujo por frame:
1. Cada observación se asigna a su *tracklet* (track del tracker, posiblemente partido en fragmentos).
2. Partición (RF-5): si la apariencia o el equipo del tracklet cambian de forma sostenida, se parte.
3. Resolución de separación (RF-6b): tras una superposición, se reasignan identidades dentro del grupo.
4. Los tracklets con identidad actualizan su identidad (galería congelada en superposición, RF-6).
5. Los tracklets sin identidad se asocian (húngaro, RF-4) a identidades no vigentes o crean una nueva
   (RF-3, RF-8) si hay evidencia y cupo.
6. Dorsal por votación con unicidad por equipo (RF-7). Estados según la tabla de la spec.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from volley_cv.identity.jersey import JerseyVotes, resolve_team_conflicts
from volley_cv.identity.settings import IdentityConfig
from volley_cv.identity.types import JerseyRead, Observation, PlayerState, Team
from volley_cv.output.schema import FrameOutput, PlayerOut

Vec = NDArray[np.float64]
INF = 1e6
VIGENTES = (PlayerState.DETECTED, PlayerState.TRACKED, PlayerState.REIDENTIFIED, PlayerState.OCCLUDED)


def _unit(v: NDArray[np.floating]) -> Vec | None:
    n = float(np.linalg.norm(v))
    return None if n == 0 or not np.isfinite(n) else np.asarray(v, dtype=np.float64) / n


def _cos_dist(a: Vec, b: Vec) -> float:
    return float(1.0 - np.dot(a, b))


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Tracklet:
    key: str
    raw: int
    obs_count: int = 0
    last_frame: int = -1
    ref: Vec | None = None
    last_emb: Vec | None = None
    last_team: Team | None = None
    center: tuple[float, float] = (0.0, 0.0)
    height: float = 1.0
    team_votes: Counter[Team] = field(default_factory=Counter)
    far_count: int = 0
    team_mismatch: int = 0
    recent: list[Vec] = field(default_factory=list)
    reads: list[JerseyRead] = field(default_factory=list)
    player: str | None = None
    wait: int = 0

    @property
    def team(self) -> Team | None:
        if not self.team_votes:
            return None
        (top, n), *rest = self.team_votes.most_common()
        return None if rest and rest[0][1] == n else top

    def confident_read(self, min_conf: float) -> int | None:
        good = [r for r in self.reads if r.confidence >= min_conf]
        if not good:
            return None
        return Counter(r.number for r in good).most_common(1)[0][0]


@dataclass
class _Identity:
    pid: str
    team: Team
    created: int
    gallery: deque[Vec]
    last_frame: int
    center: tuple[float, float]
    height: float
    state: PlayerState = PlayerState.DETECTED
    vel: tuple[float, float] = (0.0, 0.0)
    tracklet: str | None = None
    votes: JerseyVotes = field(default_factory=JerseyVotes)
    jersey: tuple[int, float] | None = None
    after_cut: bool = False

    def gallery_mean(self) -> Vec | None:
        if not self.gallery:
            return None
        return _unit(np.mean(np.stack(list(self.gallery)), axis=0))


class IdentityManager:
    def __init__(self, config: IdentityConfig | None = None) -> None:
        self.config = config or IdentityConfig()
        self._last_frame = -1
        self._tracklets: dict[str, _Tracklet] = {}
        self._raw_to_key: dict[int, str] = {}
        self._split_count: Counter[int] = Counter()
        self._identities: dict[str, _Identity] = {}
        self._next_number: Counter[Team] = Counter()
        self._partners: dict[str, set[str]] = {}
        self._since_sep: dict[str, int] = {}

    # ── API pública ──────────────────────────────────────────────────────────────
    def update(self, frame: int, observations: Sequence[Observation]) -> FrameOutput:
        if frame <= self._last_frame:
            raise ValueError(f"frame {frame} no es posterior a {self._last_frame}; usar reset() en un corte")
        self._last_frame = frame
        cfg = self.config

        obs = [o for o in observations if o.bbox[2] > o.bbox[0] and o.bbox[3] > o.bbox[1]]
        embs: list[Vec | None] = [None if o.embedding is None else _unit(o.embedding) for o in obs]
        overlapping = self._overlaps(obs)

        frame_state: dict[str, PlayerState] = {}
        observed: list[tuple[_Tracklet, Observation, int]] = []
        for i, o in enumerate(obs):
            t = self._observe_tracklet(o, embs[i], i in overlapping, frame)
            observed.append((t, o, i))

        self._resolve_separations(observed, overlapping, frame)

        # tracklets con identidad: actualizan su identidad
        for t, o, i in observed:
            if t.player is not None:
                self._update_identity(self._identities[t.player], t, o, embs[i], i in overlapping, frame)

        # tracklets sin identidad: asociación o creación
        unassigned = [(t, o, i) for t, o, i in observed if t.player is None and t.team is not None]
        self._associate(unassigned, embs, frame, frame_state)

        # estados de identidades no observadas
        observed_pids = {t.player for t, _, _ in observed if t.player is not None}
        for idn in self._identities.values():
            if idn.pid in observed_pids:
                continue
            missed = frame - idn.last_frame
            idn.state = PlayerState.OCCLUDED if missed <= cfg.lost_after else PlayerState.LOST

        self._resolve_jerseys()

        players = []
        for t, o, _ in observed:
            if t.player is None:
                continue
            idn = self._identities[t.player]
            state = frame_state.get(idn.pid, idn.state)
            idn.state = state
            number, conf = idn.jersey if idn.jersey else (None, 0.0)
            players.append(
                PlayerOut(
                    player_id=idn.pid,
                    track_id=f"track_{t.key}",
                    team_id=f"TEAM_{idn.team}",  # type: ignore[arg-type]
                    jersey_number=number,
                    jersey_confidence=round(float(conf), 4),
                    bbox=o.bbox,
                    confidence=float(min(max(o.confidence, 0.0), 1.0)),
                    state=state.value,
                )
            )
        # al frame siguiente, DETECTED/REIDENTIFIED pasan a TRACKED si se siguen observando
        for pid, st in frame_state.items():
            if st in (PlayerState.DETECTED, PlayerState.REIDENTIFIED):
                self._identities[pid].state = PlayerState.TRACKED
        return FrameOutput(frame=frame, players=players)

    def identity_states(self) -> dict[str, PlayerState]:
        """Estado actual de todas las identidades (incluye OCCLUDED/LOST, que no salen en FrameOutput)."""
        return {pid: idn.state for pid, idn in sorted(self._identities.items())}

    def reset(self) -> None:
        """Corte de escena: descarta tracklets; las identidades quedan LOST (Re-ID solo por apariencia)."""
        self._tracklets.clear()
        self._raw_to_key.clear()
        self._partners.clear()
        self._since_sep.clear()
        for idn in self._identities.values():
            idn.tracklet = None
            idn.state = PlayerState.LOST
            idn.after_cut = True

    # ── tracklets ────────────────────────────────────────────────────────────────
    def _overlaps(self, obs: Sequence[Observation]) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                if _iou(obs[i].bbox, obs[j].bbox) > self.config.overlap_iou:
                    out.setdefault(i, set()).add(j)
                    out.setdefault(j, set()).add(i)
        return out

    def _observe_tracklet(self, o: Observation, emb: Vec | None, overlap: bool, frame: int) -> _Tracklet:
        cfg = self.config
        key = self._raw_to_key.get(o.track_id)
        t = self._tracklets.get(key) if key is not None else None
        if t is None:
            t = self._new_tracklet(o.track_id)
        elif not overlap:
            if emb is not None and t.ref is not None and _cos_dist(emb, t.ref) > cfg.split_distance:
                t.far_count += 1
                t.recent.append(emb)
            else:
                t.far_count = 0
                t.recent.clear()
            team = t.team
            t.team_mismatch = t.team_mismatch + 1 if o.team and team and o.team != team else 0
            if t.far_count >= cfg.split_frames or t.team_mismatch >= cfg.team_split_frames:
                t = self._split(t)

        t.obs_count += 1
        t.last_frame = frame
        t.center, t.height = o.center, o.height
        t.last_emb = emb
        t.last_team = o.team
        if o.team is not None:
            t.team_votes[o.team] += 1
        if emb is not None:
            if t.ref is None:
                t.ref = emb
            elif not overlap and t.far_count == 0:
                t.ref = _unit(cfg.ref_ema * t.ref + (1 - cfg.ref_ema) * emb)
        if o.jersey is not None:
            t.reads.append(o.jersey)
            if t.player is not None:
                self._identities[t.player].votes.add(o.jersey)
        return t

    def _new_tracklet(self, raw: int) -> _Tracklet:
        n = self._split_count[raw]
        key = str(raw) if n == 0 else f"{raw}.{n}"
        self._split_count[raw] += 1
        t = _Tracklet(key=key, raw=raw)
        self._tracklets[key] = t
        self._raw_to_key[raw] = key
        return t

    def _split(self, old: _Tracklet) -> _Tracklet:
        if old.player is not None:
            idn = self._identities[old.player]
            if idn.tracklet == old.key:
                idn.tracklet = None
        old.player = None
        recent = old.recent
        new = self._new_tracklet(old.raw)
        new.ref = _unit(np.mean(np.stack(recent), axis=0)) if recent else None
        new.obs_count = len(recent)
        if old.last_team is not None:
            new.team_votes[old.last_team] += max(len(recent), old.team_mismatch)
        return new

    # ── superposición y separación (RF-6b) ───────────────────────────────────────
    def _resolve_separations(
        self, observed: list[tuple[_Tracklet, Observation, int]], overlapping: dict[int, set[int]], frame: int
    ) -> None:
        cfg = self.config
        pid_of = {i: t.player for t, _, i in observed}
        # acumular compañeros de superposición
        for i, others in overlapping.items():
            pi = pid_of.get(i)
            if pi is None:
                continue
            for j in others:
                pj = pid_of.get(j)
                if pj is not None:
                    self._partners.setdefault(pi, set()).add(pj)
            self._since_sep.pop(pi, None)
        # identidades separadas recientemente
        free: dict[str, tuple[_Tracklet, int]] = {
            t.player: (t, i) for t, _, i in observed if t.player is not None and i not in overlapping
        }
        window: set[str] = set()
        for pid in list(self._partners):
            if pid not in free:
                continue
            self._since_sep[pid] = self._since_sep.get(pid, 0) + 1
            if self._since_sep[pid] > cfg.separation_frames:
                self._partners.pop(pid, None)
                self._since_sep.pop(pid, None)
            else:
                window.add(pid)
        # componentes conexas entre identidades en ventana de separación
        seen: set[str] = set()
        for start in sorted(window):
            if start in seen:
                continue
            comp, stack = set(), [start]
            while stack:
                p = stack.pop()
                if p in comp:
                    continue
                comp.add(p)
                stack.extend(q for q in self._partners.get(p, ()) if q in window and q not in comp)
            seen |= comp
            if len(comp) >= 2:
                self._reassign_group(sorted(comp), free)

    def _reassign_group(self, pids: list[str], free: dict[str, tuple[_Tracklet, int]]) -> None:
        cfg = self.config
        tracklets = [free[p][0] for p in pids]
        cost = np.full((len(tracklets), len(pids)), INF)
        for a, t in enumerate(tracklets):
            for b, pid in enumerate(pids):
                idn = self._identities[pid]
                g = idn.gallery_mean()
                if t.last_team is not None and t.last_team != idn.team:
                    continue
                if g is not None and t.last_emb is not None:
                    cost[a, b] = _cos_dist(t.last_emb, g)
        current = sum(cost[a, a] for a in range(len(pids)))
        rows, cols = linear_sum_assignment(cost)
        best = float(cost[rows, cols].sum())
        if current - best <= cfg.swap_margin or best >= INF:
            return
        for a, b in zip(rows, cols, strict=True):
            t, pid = tracklets[a], pids[b]
            t.player = pid
            self._identities[pid].tracklet = t.key
            t.ref, t.far_count, t.recent = t.last_emb, 0, []

    # ── identidades ──────────────────────────────────────────────────────────────
    def _update_identity(
        self, idn: _Identity, t: _Tracklet, o: Observation, emb: Vec | None, overlap: bool, frame: int
    ) -> None:
        gap = max(frame - idn.last_frame, 1)
        if idn.last_frame >= 0:
            inst = ((o.center[0] - idn.center[0]) / gap, (o.center[1] - idn.center[1]) / gap)
            vmax = 0.3 * o.height
            inst = (float(np.clip(inst[0], -vmax, vmax)), float(np.clip(inst[1], -vmax, vmax)))
            idn.vel = (0.7 * idn.vel[0] + 0.3 * inst[0], 0.7 * idn.vel[1] + 0.3 * inst[1])
        idn.center, idn.height, idn.last_frame = o.center, o.height, frame
        idn.tracklet = t.key
        idn.after_cut = False
        if emb is not None and not overlap and t.far_count == 0:
            idn.gallery.append(emb)
        if idn.state in (PlayerState.OCCLUDED, PlayerState.LOST):
            idn.state = PlayerState.TRACKED

    def _cost(self, t: _Tracklet, idn: _Identity, frame: int) -> float:
        cfg = self.config
        missed = frame - idn.last_frame
        if idn.after_cut or missed > cfg.long_gap_frames:
            d_mot = 0.5
        else:
            px = idn.center[0] + idn.vel[0] * missed
            py = idn.center[1] + idn.vel[1] * missed
            dist_h = float(np.hypot(t.center[0] - px, t.center[1] - py)) / max(idn.height, 1.0)
            radius = min(cfg.base_radius_h + cfg.radius_growth_h * missed, cfg.max_radius_h)
            if dist_h > radius:
                return INF
            d_mot = dist_h / radius
        g = idn.gallery_mean()
        jersey = 0.0
        read = t.confident_read(cfg.jersey_min_conf)
        if read is not None and idn.jersey is not None:
            jersey = -cfg.jersey_bonus if read == idn.jersey[0] else cfg.jersey_penalty
        if t.ref is None or g is None:
            return 2 * cfg.w_motion * d_mot + jersey  # sin apariencia: más exigente
        d_app = min(_cos_dist(t.ref, x) for x in idn.gallery) if idn.gallery else _cos_dist(t.ref, g)
        return cfg.w_appearance * d_app + cfg.w_motion * d_mot + jersey

    def _associate(
        self,
        unassigned: list[tuple[_Tracklet, Observation, int]],
        embs: list[Vec | None],
        frame: int,
        frame_state: dict[str, PlayerState],
    ) -> None:
        cfg = self.config
        if not unassigned:
            return
        busy = {t.player for t in self._tracklets.values() if t.player and t.last_frame == frame}
        candidates = [
            idn
            for idn in sorted(self._identities.values(), key=lambda x: x.pid)
            if idn.pid not in busy and idn.last_frame < frame
        ]
        cost = np.full((len(unassigned), len(candidates)), INF)
        for a, (t, _, _) in enumerate(unassigned):
            for b, idn in enumerate(candidates):
                if idn.team == t.team:
                    cost[a, b] = self._cost(t, idn, frame)
        matched: set[int] = set()
        if candidates:
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols, strict=True):
                if cost[a, b] < cfg.accept_cost:
                    t, o, i = unassigned[a]
                    idn = candidates[b]
                    self._link(t, idn)
                    self._update_identity(idn, t, o, embs[i], False, frame)
                    frame_state[idn.pid] = PlayerState.REIDENTIFIED
                    matched.add(a)
        for a, (t, o, i) in enumerate(unassigned):
            if a in matched or t.obs_count < cfg.confirm_frames:
                continue
            team = t.team
            assert team is not None
            same_team = [cost[a, b] for b, idn in enumerate(candidates) if idn.team == team]
            ambiguous = bool(same_team) and min(same_team) < cfg.new_identity_cost
            if ambiguous:
                t.wait += 1
                if t.wait <= cfg.max_wait_frames:
                    continue
            if self._active(team) >= cfg.roster_size:
                continue
            idn = self._create(team, t, o, embs[i], frame)
            frame_state[idn.pid] = PlayerState.DETECTED

    def _active(self, team: Team) -> int:
        return sum(1 for i in self._identities.values() if i.team == team and i.state in VIGENTES)

    def _create(self, team: Team, t: _Tracklet, o: Observation, emb: Vec | None, frame: int) -> _Identity:
        self._next_number[team] += 1
        pid = f"TEAM_{team}_PLAYER_{self._next_number[team]:02d}"
        idn = _Identity(
            pid=pid,
            team=team,
            created=frame,
            gallery=deque(maxlen=self.config.gallery_size),
            last_frame=-1,
            center=o.center,
            height=o.height,
        )
        if t.ref is not None:
            idn.gallery.append(t.ref)
        self._identities[pid] = idn
        self._link(t, idn)
        self._update_identity(idn, t, o, emb, False, frame)
        idn.state = PlayerState.DETECTED
        return idn

    def _link(self, t: _Tracklet, idn: _Identity) -> None:
        if idn.tracklet is not None and idn.tracklet in self._tracklets:
            old = self._tracklets[idn.tracklet]
            if old.player == idn.pid:
                old.player = None
        t.player = idn.pid
        idn.tracklet = t.key
        for r in t.reads:
            idn.votes.add(r)

    def _resolve_jerseys(self) -> None:
        for team in Team:
            cands = {
                pid: idn.votes.candidate(self.config)
                for pid, idn in sorted(self._identities.items())
                if idn.team == team
            }
            for pid, res in resolve_team_conflicts(cands).items():
                self._identities[pid].jersey = res
