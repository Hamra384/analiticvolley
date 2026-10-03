"""BallTracker (SPEC-005 RF-2…RF-5): Kalman de aceleración constante en píxeles y estados por frame.

Una sola pelota por tramo. Por frame se asocia a lo sumo un candidato: el más cercano a la predicción,
dentro de un radio que crece con los frames sin detección. Sin detección se predice hasta `max_predict` frames
(`PREDICTED`, nunca `DETECTED`/`TRACKED`, RN-1); después, `LOST` sin posición.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from volley_cv.output.schema import BallStateName

# estado: [x, y, vx, vy, ax, ay]; paso de 1 frame
_F = np.eye(6)
_F[0, 2] = _F[1, 3] = 1.0
_F[0, 4] = _F[1, 5] = 0.5
_F[2, 4] = _F[3, 5] = 1.0
_H = np.zeros((2, 6))
_H[0, 0] = _H[1, 1] = 1.0


@dataclass(frozen=True)
class BallTrackerConfig:
    max_predict: int = 8  # frames de predicción antes de LOST (~0,27 s a 30 FPS)
    start_conf: float = 0.25  # confianza mínima para empezar un track sin evidencia previa
    gate_px: float = 60.0  # radio de asociación con el track establecido
    gate_init_px: float = 150.0  # radio mientras la velocidad todavía no está estimada
    gate_growth_px: float = 15.0  # crecimiento del radio por frame sin detección
    gate_max_px: float = 400.0
    init_frames: int = 3  # detecciones hasta considerar estimada la velocidad
    meas_std: float = 2.0  # ruido de medición (px)
    accel_std: float = 0.5  # ruido de proceso en la aceleración (px/frame²)
    conf_decay: float = 0.8  # factor de confianza por frame predicho
    # RF-2b: una detección que no se mueve durante `static_frames` es un señuelo (pelota de repuesto
    # en la mano de un alcanzapelotas, pelota en el piso): se ignora mientras siga quieta
    static_frames: int = 45  # 1,5 s a 30 FPS
    static_radius_px: float = 6.0
    static_forget: int = 15  # frames sin verla para olvidar el señuelo
    reacquire_frames: int = 60  # tras LOST, REACQUIRED solo dentro de este plazo (después es un track nuevo)
    max_candidates: int = 10  # los de mayor confianza (costo acotado, RNF-1)


@dataclass(frozen=True)
class BallState:
    track_id: str
    position: tuple[float, float] | None
    confidence: float
    state: BallStateName


class BallTracker:
    def __init__(self, config: BallTrackerConfig | None = None) -> None:
        self.config = config or BallTrackerConfig()
        self._n_tracks = 0
        self._x: NDArray[np.float64] | None = None
        self._p: NDArray[np.float64] = np.eye(6)
        self._missed = 0
        self._hits = 0
        self._conf = 0.0
        self._lost = False  # el track actual pasó por LOST (la próxima detección compatible es REACQUIRED)
        self._frame = 0
        self._spots: list[list[float]] = []  # [x, y, frames quieta, último frame visto] (RF-2b)
        # ruido de proceso: un salto de aceleración afecta posición (½), velocidad (1) y aceleración (1)
        self._q = np.diag([0.25, 0.25, 1.0, 1.0, 1.0, 1.0]) * self.config.accel_std**2
        self._r = np.eye(2) * self.config.meas_std**2

    def reset(self) -> None:
        """Corte de escena (RF-5): se descarta el track."""
        self._spots.clear()
        self._x = None
        self._missed = self._hits = 0
        self._lost = False

    def update(self, candidates: NDArray[np.float32]) -> BallState:
        cfg = self.config
        cands = self._drop_decoys(np.asarray(candidates, dtype=np.float64).reshape(-1, 5))
        if self._x is None:
            return self._start(cands)
        if self._lost:
            # revisión S5b M1: en LOST la posición queda congelada (no se extrapola sin evidencia), el radio
            # no crece y solo una detección confiable dentro del plazo es REACQUIRED; si no, track nuevo
            self._missed += 1
            if self._missed - cfg.max_predict > cfg.reacquire_frames:
                return self._start(cands)
            cands = cands[cands[:, 4] >= cfg.start_conf] if len(cands) else cands
            radius = min(cfg.gate_px + cfg.gate_growth_px * cfg.max_predict, cfg.gate_max_px)
        else:
            self._predict()
            radius = min(
                (cfg.gate_init_px if self._hits < cfg.init_frames else cfg.gate_px)
                + cfg.gate_growth_px * self._missed,
                cfg.gate_max_px,
            )
        assert self._x is not None
        pred = self._x[:2]
        best = None
        if len(cands):
            d = np.hypot(cands[:, 0] - pred[0], cands[:, 1] - pred[1])
            k = int(np.argmin(d))
            if d[k] <= radius:
                best = cands[k]
        if best is not None:
            self._correct(best[:2])
            state: BallStateName = "REACQUIRED" if self._lost else "TRACKED"
            self._missed, self._lost, self._conf = 0, False, float(best[4])
            self._hits += 1
            return self._out(state)
        if not self._lost:
            self._missed += 1
        if self._missed <= cfg.max_predict and not self._lost:
            return BallState(
                self._id,
                (float(pred[0]), float(pred[1])),
                self._conf * cfg.conf_decay**self._missed,
                "PREDICTED",
            )
        if not self._lost:
            self._lost = True
            self._x[2:] = 0.0  # se congela la última predicción
        # LOST: una detección incompatible pero confiable empieza un track nuevo
        fresh = cands[cands[:, 4] >= cfg.start_conf] if len(cands) else cands
        if len(fresh):
            return self._start(fresh)
        return BallState(self._id, None, 0.0, "LOST")

    # ── internos ──────────────────────────────────────────────────────────────────
    def _drop_decoys(self, cands: NDArray[np.float64]) -> NDArray[np.float64]:
        """RF-2b: actualiza los lugares donde hay una detección quieta y descarta las que ya son señuelos.

        El lugar es un ancla fija (la primera posición): una pelota que se mueve, aunque sea lento, sale del
        radio y no se marca (revisión S5b M2). Cada lugar suma a lo sumo una vez por frame (B3).
        """
        cfg = self.config
        self._frame += 1
        if len(cands) > cfg.max_candidates:  # B4: costo acotado
            cands = cands[np.argsort(-cands[:, 4])[: cfg.max_candidates]]
        keep = []
        for c in cands:
            r = max(cfg.static_radius_px, 0.6 * float(c[2]))  # tolera el temblor de una pelota en la mano
            spot = next((s for s in self._spots if np.hypot(c[0] - s[0], c[1] - s[1]) <= r), None)
            if spot is None:
                spot = [float(c[0]), float(c[1]), 0.0, float(self._frame - 1)]
                self._spots.append(spot)
            if spot[3] < self._frame:
                spot[2] += 1
                spot[3] = float(self._frame)
            keep.append(spot[2] < cfg.static_frames)
        self._spots = [s for s in self._spots if s[3] >= self._frame - cfg.static_forget]
        return cands[np.array(keep, dtype=bool)] if len(cands) else cands

    @property
    def _id(self) -> str:
        return f"ball_{max(self._n_tracks, 1)}"

    def _start(self, cands: NDArray[np.float64]) -> BallState:
        cands = cands[cands[:, 4] >= self.config.start_conf] if len(cands) else cands
        if not len(cands):
            return BallState(self._id, None, 0.0, "LOST")
        c = cands[int(np.argmax(cands[:, 4]))]
        self._n_tracks += 1
        self._x = np.array([c[0], c[1], 0.0, 0.0, 0.0, 0.0])
        self._p = np.diag([self.config.meas_std**2] * 2 + [50.0**2] * 2 + [5.0**2] * 2)
        self._missed, self._hits, self._lost, self._conf = 0, 1, False, float(c[4])
        return self._out("DETECTED")

    def _predict(self) -> None:
        assert self._x is not None
        self._x = _F @ self._x
        self._p = _F @ self._p @ _F.T + self._q

    def _correct(self, z: NDArray[np.float64]) -> None:
        assert self._x is not None
        y = z - _H @ self._x
        s = _H @ self._p @ _H.T + self._r
        k = self._p @ _H.T @ np.linalg.inv(s)
        self._x = self._x + k @ y
        self._p = (np.eye(6) - k @ _H) @ self._p

    def _out(self, state: BallStateName) -> BallState:
        assert self._x is not None
        return BallState(self._id, (float(self._x[0]), float(self._x[1])), self._conf, state)
