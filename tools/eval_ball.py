"""Métricas proxy de pelota sobre los 10 clips de evaluación (SPEC-005 RF-8; métricas de SPIKE-002).

Mismo tracker de producción (BallTracker) para todos los detectores:
- P1 cobertura: frames con la pelota detectada y asociada (DETECTED/TRACKED/REACQUIRED) / frames del clip.
- P2 plausibilidad: frames en los que el track usó un candidato / frames con al menos un candidato.
- P3 saltos: pasos con posición detectada > 5 % del ancho del frame entre frames consecutivos.
- P5 FPS: solo detección.
- P4 (precisión visual) es manual: se guarda una hoja con 12 recortes al azar por clip (marca en el centro).

Uso: uv run --extra ml python -m tools.eval_ball --name coco            (YOLOv8m COCO, clase 32)
     uv run --extra ml python -m tools.eval_ball --name ball_v1 --weights models/ball_v1.pt
Salida: <datos>/cache/reports/ball_eval_<name>.json y ball_p4_<name>_<clip>.jpg
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import cv2
import numpy as np

from volley_cv.adapters import YoloBallDetector
from volley_cv.ball import BallTracker
from volley_cv.config import PROJECT_ROOT, load_clips, load_settings
from volley_cv.video.source import read_frames

SEEN = ("DETECTED", "TRACKED", "REACQUIRED")


def crop(frame: np.ndarray, x: float, y: float, s: int = 60) -> np.ndarray:
    """Recorte centrado en (x, y), con relleno negro en los bordes (la marca queda siempre en el centro)."""
    pad = cv2.copyMakeBorder(frame, s, s, s, s, cv2.BORDER_CONSTANT)
    xi, yi = round(x) + s, round(y) + s
    c = pad[yi - s : yi + s, xi - s : xi + s].copy()
    cv2.circle(c, (s, s), 14, (0, 0, 255), 1)
    return c


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", help="pesos propios (1 clase), relativos al directorio de datos")
    args = ap.parse_args(argv)
    data = load_settings().data_dir
    catalog = load_clips(PROJECT_ROOT / "configs" / "eval" / "clips.yaml")
    if args.weights:
        det = YoloBallDetector(data / args.weights)
    else:
        det = YoloBallDetector(data / "yolov8m.pt", classes=[32])
    out_dir = data / "cache" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    results = []
    for clip in catalog.clips:
        tracker = BallTracker()
        states, used, with_cand, jumps, t_det, n = [], 0, 0, 0, 0.0, 0
        positions: list[tuple[float, float] | None] = []
        prev: tuple[float, float] | None = None
        width = 0
        for _, frame in read_frames(catalog.video_path(clip, data), clip.start_s, clip.end_s):
            width = frame.shape[1]
            t0 = time.perf_counter()
            cands = det.detect(frame)
            t_det += time.perf_counter() - t0
            st = tracker.update(cands)
            n += 1
            states.append(st.state)
            if len(cands):
                with_cand += 1
            seen = st.state in SEEN
            if seen:
                used += 1
                p = st.position
                if (
                    prev is not None
                    and p is not None
                    and np.hypot(p[0] - prev[0], p[1] - prev[1]) > 0.05 * width
                ):
                    jumps += 1
            prev = st.position if seen else None
            positions.append(st.position if seen else None)
        seen_idx = [k for k, p in enumerate(positions) if p is not None]
        sample = sorted(rng.sample(seen_idx, min(12, len(seen_idx))))
        tiles = []
        for k, (_, frame) in enumerate(read_frames(catalog.video_path(clip, data), clip.start_s, clip.end_s)):
            if k in sample:
                p = positions[k]
                assert p is not None
                c = crop(frame, *p)
                cv2.putText(c, str(k), (2, 12), 0, 0.4, (0, 255, 255), 1)
                tiles.append(c)
        if tiles:
            while len(tiles) % 6:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[r : r + 6]) for r in range(0, len(tiles), 6)])
            cv2.imwrite(str(out_dir / f"ball_p4_{args.name}_{clip.id}.jpg"), sheet)
        r = {
            "clip": clip.id,
            "P1": round(used / max(n, 1), 3),
            "P2": round(used / max(with_cand, 1), 3),
            "P3": jumps,
            "P5_fps": round(n / max(t_det, 1e-9), 1),
            "states": {s: states.count(s) for s in sorted(set(states))},
        }
        results.append(r)
        print(json.dumps(r), flush=True)
    p1 = float(np.median([r["P1"] for r in results]))
    summary = {"name": args.name, "P1_median": p1, "P1_min": min(r["P1"] for r in results), "clips": results}
    (out_dir / f"ball_eval_{args.name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"name": args.name, "P1_median": p1}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
