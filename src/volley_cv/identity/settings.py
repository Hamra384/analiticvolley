"""Parámetros del IdentityManager (SPEC-001). Valores iniciales; se ajustan con evaluación (S6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityConfig:
    # creación y estados
    confirm_frames: int = 5  # observaciones de un tracklet antes de crear identidad (RF-3)
    lost_after: int = 15  # frames sin observación: OCCLUDED -> LOST
    roster_size: int = 6  # identidades vigentes máximas por equipo (RF-8)
    max_wait_frames: int = 30  # tracklet ambiguo: frames de espera antes de decidir
    team_min_share: float = 0.8  # proporción mínima de votos del mismo equipo para crear identidad (RF-3b)

    # asociación (RF-4); distancias de apariencia = coseno (0..2), movimiento en alturas de jugador
    w_appearance: float = 0.6
    w_motion: float = 0.4
    accept_cost: float = 0.30  # costo máximo para asociar a una identidad existente
    new_identity_cost: float = 0.40  # costo mínimo a todas las candidatas para crear identidad nueva
    base_radius_h: float = 1.0  # radio de búsqueda inicial (alturas de jugador)
    radius_growth_h: float = 0.2  # crecimiento del radio por frame perdido
    max_radius_h: float = 6.0
    long_gap_frames: int = 90  # tras este hueco el movimiento deja de restringir (solo apariencia)
    # Re-ID tras hueco largo o corte (sin restricción de movimiento): distancia de apariencia máxima (RF-4d)
    appearance_only_max_dist: float = 0.08
    reid_min_obs_without_embedding: int = 3  # sin apariencia, un tracklet de 1-2 frames no re-identifica
    jersey_bonus: float = 0.25
    jersey_penalty: float = 0.5

    # apariencia
    gallery_size: int = 30
    ref_ema: float = 0.9  # media móvil de la referencia de un tracklet

    # partición de tracklets (RF-5)
    # SPIKE-003 (histograma de color, A2/K2): mismo jugador p90 = 0,14; compañeros distintos mediana 0,25-0,37
    split_distance: float = 0.2
    split_frames: int = 8
    team_split_frames: int = 3

    # superposición y separación (RF-6, RF-6b)
    overlap_iou: float = 0.3
    separation_frames: int = 3
    swap_margin: float = 0.1

    # dorsal (RF-7)
    jersey_min_reads: int = 3
    jersey_min_conf: float = 0.8
    jersey_min_share: float = 0.7
    jersey_split_reads: int = 3  # lecturas consecutivas de otro número que parten el tracklet (SPEC-003 RF-5)
