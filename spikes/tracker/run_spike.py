"""SPIKE-001: compara 4 trackers sobre las mismas detecciones (docs/spikes/001-tracker.md).

Uso: uv run --extra ml python -m spikes.tracker.run_spike [--clips A1 A2 ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spikes.common import ClipInfo, cache_dir, clips, data_dir, frames

PLAYER_MIN_H = 0.07  # fracción del alto del frame (excluye público)
MIN_RUN = 15  # frames sostenidos antes y después de un cambio de color
SMOOTH = 9


def detect(info: ClipInfo) -> list[np.ndarray]:
    path = cache_dir("dets") / f"{info.clip.id}_yolov8m_1280.npz"
    if path.is_file():
        z = np.load(path)
        return [z[f"f{i}"] for i in range(int(z["n"]))]
    from ultralytics import YOLO

    model = YOLO(str(data_dir() / "yolov8m.pt"))
    out = []
    for _, frame in frames(info):
        r = model.predict(frame, classes=[0], imgsz=1280, conf=0.10, verbose=False)[0]
        b = r.boxes
        d = np.concatenate(
            [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], axis=1
        )
        out.append(d.astype(np.float32))
    np.savez_compressed(path, n=len(out), **{f"f{i}": d for i, d in enumerate(out)})
    return out


def make_tracker(name: str) -> Any:
    import torch
    from boxmot import BotSort, ByteTrack, DeepOcSort

    reid = data_dir() / "weights" / "osnet_x0_25_msmt17.pt"
    reid.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "T1_bytetrack":
        return ByteTrack(min_conf=0.1, track_thresh=0.4, track_buffer=30, frame_rate=30)
    if name == "T2_botsort_cmc":
        return BotSort(use_cmc=True, use_embeddings=False, reid_weights=reid, device=dev, half=False)
    if name == "T3_botsort_cmc_reid":
        return BotSort(use_cmc=True, use_embeddings=True, reid_weights=reid, device=dev, half=False)
    if name == "T4_deepocsort_reid":
        return DeepOcSort(reid_weights=reid, device=dev, half=False)
    raise ValueError(name)


def torso_lab(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    tx1, tx2 = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    ty1, ty2 = int(y1 + 0.2 * h), int(y1 + 0.45 * h)
    crop = frame[max(ty1, 0) : ty2, max(tx1, 0) : tx2]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)


def color_flips(labels: list[int]) -> int:
    if len(labels) < 2 * MIN_RUN:
        return 0
    arr = np.array(labels)
    sm = np.array(
        [np.bincount(arr[max(0, i - SMOOTH // 2) : i + SMOOTH // 2 + 1]).argmax() for i in range(len(arr))]
    )
    runs = []  # (label, length)
    for v in sm:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    long_runs = [r for r in runs if r[1] >= MIN_RUN]
    return sum(1 for a, b in itertools.pairwise(long_runs) if a[0] != b[0])


def run(info: ClipInfo, dets: list[np.ndarray], tracker_name: str) -> dict[str, Any]:
    tracker = make_tracker(tracker_name)
    min_h = PLAYER_MIN_H * info.height
    per_track_colors: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    spans: dict[int, list[int]] = {}
    per_frame_count = []
    t_track = 0.0
    tracks_out = []
    for i, frame in frames(info):
        d = dets[i]
        t0 = time.perf_counter()
        tr = tracker.update(d if len(d) else np.empty((0, 6), np.float32), frame)
        t_track += time.perf_counter() - t0
        n = 0
        for row in tr:
            box, tid = row[:4], int(row[4])
            if box[3] - box[1] < min_h:
                continue
            n += 1
            spans.setdefault(tid, [i, i])[1] = i
            c = torso_lab(frame, box)
            if c is not None:
                per_track_colors[tid].append((i, c))
            tracks_out.append([i, tid, *box.tolist()])
        per_frame_count.append(n)

    all_colors = np.array([c for v in per_track_colors.values() for _, c in v], np.float32)
    flips_total, flipped_tracks = 0, []
    if len(all_colors) >= 3:
        cv2.setRNGSeed(42)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, _, centers = cv2.kmeans(all_colors, 3, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        for tid, seq in per_track_colors.items():
            labels = [int(np.argmin(np.linalg.norm(centers - c, axis=1))) for _, c in seq]
            f = color_flips(labels)
            if f:
                flips_total += f
                flipped_tracks.append(tid)
    simult = float(np.median(per_frame_count)) if per_frame_count else 0.0
    lifetimes = [(e - s + 1) / info.n_frames for s, e in spans.values()]
    out_path = cache_dir("tracks") / f"{info.clip.id}_{tracker_name}.npy"
    np.save(out_path, np.array(tracks_out, np.float32))
    return {
        "clip": info.clip.id,
        "tracker": tracker_name,
        "M1_color_flips": flips_total,
        "flipped_tracks": flipped_tracks,
        "M2_fragmentation": round(len(spans) / simult, 2) if simult else None,
        "M3_median_life": round(float(np.median(lifetimes)), 3) if lifetimes else None,
        "M4_fps": round(len(per_frame_count) / t_track, 1) if t_track else None,
        "ids": len(spans),
        "median_simultaneous": simult,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    ap.add_argument(
        "--trackers",
        nargs="*",
        default=["T1_bytetrack", "T2_botsort_cmc", "T3_botsort_cmc_reid", "T4_deepocsort_reid"],
    )
    args = ap.parse_args()
    results = []
    for info in clips(args.clips):
        dets = detect(info)
        for t in args.trackers:
            r = run(info, dets, t)
            print(json.dumps(r), flush=True)
            results.append(r)
    out = Path(cache_dir("reports")) / "spike001_results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
