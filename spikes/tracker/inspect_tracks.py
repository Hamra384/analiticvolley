"""Tira de recortes de un track a lo largo del tiempo, para validar a ojo la métrica M1.

Uso: uv run --extra ml python -m spikes.tracker.inspect_tracks <clip> <tracker> <tid> [<tid> ...]
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from spikes.common import cache_dir, clips, frames


def main() -> None:
    clip_id, tracker, tids = sys.argv[1], sys.argv[2], [int(t) for t in sys.argv[3:]]
    info = clips([clip_id])[0]
    tr = np.load(cache_dir("tracks") / f"{clip_id}_{tracker}.npy")
    wanted: dict[int, list[int]] = {}
    for tid in tids:
        rows = tr[tr[:, 1] == tid]
        idx = np.linspace(0, len(rows) - 1, 10).astype(int)
        wanted[tid] = [int(rows[k, 0]) for k in idx]
    need = {f for v in wanted.values() for f in v}
    crops: dict[tuple[int, int], np.ndarray] = {}
    for i, frame in frames(info):
        if i not in need:
            continue
        for tid, fs in wanted.items():
            if i in fs:
                r = tr[(tr[:, 0] == i) & (tr[:, 1] == tid)][0]
                x1, y1, x2, y2 = (int(v) for v in r[2:6])
                c = cv2.resize(frame[max(y1, 0) : y2, max(x1, 0) : x2], (60, 140))
                cv2.putText(c, str(i), (2, 12), 0, 0.4, (0, 255, 255), 1)
                crops[(tid, i)] = c
    rows_img = []
    for tid, fs in wanted.items():
        tiles = [crops.get((tid, f), np.zeros((140, 60, 3), np.uint8)) for f in fs]
        row = np.hstack(tiles)
        label = np.zeros((140, 50, 3), np.uint8)
        cv2.putText(label, f"#{tid}", (2, 70), 0, 0.5, (255, 255, 255), 1)
        rows_img.append(np.hstack([label, row]))
    out = cache_dir("reports") / f"inspect_{clip_id}_{tracker}.jpg"
    cv2.imwrite(str(out), np.vstack(rows_img))
    print(out)


if __name__ == "__main__":
    main()
