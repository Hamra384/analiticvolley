"""Auditoría visual M5 (SPIKE-001): los 8 tracks de jugador más largos, 10 recortes equiespaciados cada uno.

Uso: uv run --extra ml python -m spikes.tracker.audit_tracks <clip> <tracker> [<tracker> ...]
Salida: <data>/cache/reports/audit_<clip>_<tracker>.jpg
"""

from __future__ import annotations

import json
import random
import sys

import cv2
import numpy as np

from spikes.common import cache_dir, clips, frames

N_TRACKS, N_CROPS = 8, 10


def main() -> None:
    clip_id, trackers = sys.argv[1], sys.argv[2:]
    info = clips([clip_id])[0]
    plan: dict[str, dict[int, list[int]]] = {}
    data: dict[str, np.ndarray] = {}
    for t in trackers:
        tr = np.load(cache_dir("tracks") / f"{clip_id}_{t}.npy")
        data[t] = tr
        ids, counts = np.unique(tr[:, 1], return_counts=True)
        longest = [int(i) for i in ids[np.argsort(-counts)][:N_TRACKS]]
        plan[t] = {}
        for tid in longest:
            rows = tr[tr[:, 1] == tid]
            plan[t][tid] = [int(rows[k, 0]) for k in np.linspace(0, len(rows) - 1, N_CROPS).astype(int)]
    need = {f for p in plan.values() for fs in p.values() for f in fs}
    crops: dict[tuple[str, int, int], np.ndarray] = {}
    for i, frame in frames(info):
        if i not in need:
            continue
        for t, p in plan.items():
            for tid, fs in p.items():
                if i in fs:
                    r = data[t][(data[t][:, 0] == i) & (data[t][:, 1] == tid)][0]
                    x1, y1, x2, y2 = (int(v) for v in r[2:6])
                    c = cv2.resize(frame[max(y1, 0) : y2, max(x1, 0) : x2], (56, 130))
                    crops[(t, tid, i)] = c
    # auditoría ciega: nombres al azar; el mapeo se lee recién después de puntuar
    letters = ["X", "Y", "Z", "W"][: len(plan)]
    random.SystemRandom().shuffle(letters)
    mapping = dict(zip(plan, letters, strict=True))
    (cache_dir("reports") / f"audit_{clip_id}_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    for t, p in plan.items():
        rows_img = []
        for n, (tid, fs) in enumerate(p.items()):
            tiles = [crops.get((t, tid, f), np.zeros((130, 56, 3), np.uint8)) for f in fs]
            label = np.zeros((130, 30, 3), np.uint8)
            cv2.putText(label, str(n + 1), (6, 70), 0, 0.6, (255, 255, 255), 1)
            rows_img.append(np.hstack([label, *tiles]))
        out = cache_dir("reports") / f"audit_{clip_id}_{mapping[t]}.jpg"
        cv2.imwrite(str(out), np.vstack(rows_img))
        print(out)


if __name__ == "__main__":
    main()
