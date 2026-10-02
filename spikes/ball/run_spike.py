"""SPIKE-002: detectores de pelota + enlazador temporal común (docs/spikes/002-pelota.md).

Uso: uv run --extra ml python -m spikes.ball.run_spike [--clips K1 ...] [--detectors B1 B2 B3]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from typing import Any

import cv2
import numpy as np

from spikes.common import ClipInfo, cache_dir, clips, data_dir, frames

# Candidato: (x, y, score)


def yolo_candidates(info: ClipInfo, weights: str) -> tuple[list[np.ndarray], float]:
    from ultralytics import YOLO

    model = YOLO(str(data_dir() / weights))
    out, t = [], 0.0
    for _, frame in frames(info):
        t0 = time.perf_counter()
        r = model.predict(frame, classes=[32], imgsz=1280, conf=0.05, verbose=False)[0]
        t += time.perf_counter() - t0
        b = r.boxes
        xy = b.xywh.cpu().numpy()[:, :2]
        out.append(np.concatenate([xy, b.conf.cpu().numpy()[:, None]], axis=1).astype(np.float32))
    return out, len(out) / t


def classic_candidates(info: ClipInfo) -> tuple[list[np.ndarray], float]:
    """Diferencia de 3 frames + color (amarillo/azul de la pelota) + tamaño y circularidad."""
    h = info.height
    scale = (h / 1080) ** 2
    a_min, a_max = 25 * scale, 900 * scale
    buf: list[np.ndarray] = []
    out: list[np.ndarray] = []
    t = 0.0
    for _, frame in frames(info):
        t0 = time.perf_counter()
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        buf.append(g)
        if len(buf) < 3:
            out.append(np.empty((0, 3), np.float32))
            t += time.perf_counter() - t0
            continue
        buf = buf[-3:]
        d1 = cv2.absdiff(buf[1], buf[0])
        d2 = cv2.absdiff(buf[2], buf[1])
        motion = cv2.bitwise_and(
            cv2.threshold(d1, 18, 255, cv2.THRESH_BINARY)[1], cv2.threshold(d2, 18, 255, cv2.THRESH_BINARY)[1]
        )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (18, 90, 120), (38, 255, 255))
        blue = cv2.inRange(hsv, (95, 120, 80), (125, 255, 255))
        mask = cv2.bitwise_and(motion, cv2.bitwise_or(yellow, blue))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        mask[: int(0.07 * h)] = 0  # marcador sobreimpreso
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        cands = []
        for k in range(1, n):
            _, _, w, hh, area = stats[k]
            if not (a_min <= area <= a_max) or w == 0 or hh == 0:
                continue
            aspect = min(w, hh) / max(w, hh)
            fill = area / (w * hh)
            if aspect < 0.5 or fill < 0.4:
                continue
            cands.append((cents[k][0], cents[k][1], aspect * fill))
        out.append(np.array(cands, np.float32).reshape(-1, 3))
        t += time.perf_counter() - t0
    return out, len(out) / t


def link(cands: list[np.ndarray], width: int) -> tuple[list[tuple[float, float] | None], list[bool]]:
    """Enlazador simple: velocidad constante + gating creciente; re-inicio con confirmación de 2 frames."""
    pos: np.ndarray | None = None
    vel = np.zeros(2)
    missed = 0
    track: list[tuple[float, float] | None] = []
    linked: list[bool] = []
    pending: np.ndarray | None = None
    for c in cands:
        hit = None
        if pos is not None:
            pred = pos + vel
            gate = min(0.04 * width + 0.01 * width * missed, 0.15 * width)
            if len(c):
                d = np.linalg.norm(c[:, :2] - pred, axis=1)
                j = int(np.argmin(d))
                if d[j] <= gate:
                    hit = c[j, :2]
            if hit is not None:
                vel = 0.6 * (hit - pos) + 0.4 * vel if missed == 0 else (hit - pos) / (missed + 1)
                pos, missed = hit, 0
                track.append((float(hit[0]), float(hit[1])))
                linked.append(True)
                continue
            missed += 1
            pos = pred
            if missed > 10:
                pos, vel, missed = None, np.zeros(2), 0
            track.append(None)
            linked.append(False)
            continue
        # sin track: confirmar con dos detecciones consistentes
        best = c[int(np.argmax(c[:, 2])), :2] if len(c) else None
        if best is not None and pending is not None and np.linalg.norm(best - pending) <= 0.05 * width:
            pos, vel, missed = best, best - pending, 0
            track.append((float(best[0]), float(best[1])))
            linked.append(True)
        else:
            track.append(None)
            linked.append(False)
        pending = best
    return track, linked


def metrics(cands: list[np.ndarray], track: list[Any], linked: list[bool], width: int) -> dict[str, Any]:
    n = len(cands)
    frames_with_cand = sum(1 for c in cands if len(c))
    pts = [(i, p) for i, p in enumerate(track) if p is not None]
    jumps = sum(
        1
        for (i, a), (j, b) in itertools.pairwise(pts)
        if j == i + 1 and np.hypot(b[0] - a[0], b[1] - a[1]) > 0.05 * width
    )
    return {
        "P1_coverage": round(sum(linked) / n, 3) if n else None,
        "P2_plausibility": round(sum(linked) / frames_with_cand, 3) if frames_with_cand else None,
        "P3_jumps": jumps,
        "frames_with_candidate": round(frames_with_cand / n, 3) if n else None,
    }


def save_samples(info: ClipInfo, det: str, track: list[Any], rng: random.Random) -> None:
    """12 recortes al azar de posiciones enlazadas, para la verificación visual P4."""
    idx = sorted(
        rng.sample(
            [i for i, p in enumerate(track) if p is not None], k=min(12, sum(p is not None for p in track))
        )
    )
    tiles = []
    for i, frame in frames(info):
        if i in idx:
            x, y = track[i]
            r = 50
            pad = cv2.copyMakeBorder(frame, r, r, r, r, cv2.BORDER_CONSTANT)
            crop = pad[int(y) : int(y) + 2 * r, int(x) : int(x) + 2 * r].copy()
            crop = cv2.resize(crop, (120, 120), interpolation=cv2.INTER_NEAREST)
            cv2.circle(crop, (60, 60), 14, (0, 0, 255), 1)
            cv2.putText(crop, str(i), (2, 12), 0, 0.4, (0, 255, 255), 1)
            tiles.append(crop)
    while len(tiles) < 12:
        tiles.append(np.zeros((120, 120, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[k : k + 6]) for k in (0, 6)])
    cv2.imwrite(str(cache_dir("reports", "ball") / f"{info.clip.id}_{det}.jpg"), grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--detectors", nargs="*", default=["B1", "B2", "B3"])
    args = ap.parse_args()
    rng = random.Random(42)
    results = []
    for info in clips(args.clips):
        for det in args.detectors:
            if det == "B1":
                cands, fps = yolo_candidates(info, "yolov8m.pt")
            elif det == "B2":
                cands, fps = yolo_candidates(info, "yolov8l.pt")
            else:
                cands, fps = classic_candidates(info)
            track, linked = link(cands, info.width)
            r = {
                "clip": info.clip.id,
                "detector": det,
                **metrics(cands, track, linked, info.width),
                "P5_fps": round(fps, 1),
            }
            save_samples(info, det, track, rng)
            print(json.dumps(r), flush=True)
            results.append(r)
    out = cache_dir("reports") / "spike002_results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
