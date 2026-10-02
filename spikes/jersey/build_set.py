"""SPIKE-004: arma el set de recortes de torso para etiquetar números a mano (fuera del repo).

Toma tracks de ByteTrack cacheados, personas grandes (altura >= 18 % del frame), un recorte cada 15 frames por
track. Guarda <datos>/cache/jersey/crops/<id>.png, meta.json y hojas numeradas sheet_XX.jpg para etiquetar.

Uso: uv run --extra ml python -m spikes.jersey.build_set
"""

from __future__ import annotations

import json
import random

import cv2
import numpy as np

from spikes.common import cache_dir, clips, frames

MIN_H, EVERY, PER_CLIP = 0.18, 15, 45


def torso(frame: np.ndarray, box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    return frame[
        max(int(y1 + 0.12 * h), 0) : int(y1 + 0.62 * h), max(int(x1 - 0.05 * w), 0) : int(x2 + 0.05 * w)
    ]


def main() -> None:
    rng = random.Random(4)
    out = cache_dir("jersey", "crops")
    meta = []
    for info in clips(["A2", "A3", "K2", "K5"]):
        tr = np.load(cache_dir("tracks") / f"{info.clip.id}_T1_bytetrack.npy")
        big = tr[(tr[:, 5] - tr[:, 3]) >= MIN_H * info.height]
        cand = [r for r in big if int(r[0]) % EVERY == 0]
        cand = rng.sample(cand, min(PER_CLIP, len(cand)))
        want: dict[int, list[np.ndarray]] = {}
        for r in cand:
            want.setdefault(int(r[0]), []).append(r)
        for i, frame in frames(info):
            for r in want.get(i, []):
                crop = torso(frame, r[2:6].tolist())
                if crop.size == 0:
                    continue
                cid = f"{info.clip.id}_{i:03d}_{int(r[1])}"
                cv2.imwrite(str(out / f"{cid}.png"), crop)
                meta.append(
                    {
                        "id": cid,
                        "clip": info.clip.id,
                        "frame": i,
                        "track": int(r[1]),
                        "box": [round(float(v), 1) for v in r[2:6]],
                        "h": int(crop.shape[0]),
                    }
                )
    (cache_dir("jersey") / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # hojas de 8x5 con índice, para etiquetar
    for s in range(0, len(meta), 40):
        tiles = []
        for k, m in enumerate(meta[s : s + 40]):
            img = cv2.imread(str(out / f"{m['id']}.png"))
            img = cv2.resize(img, (120, 120))
            cv2.rectangle(img, (0, 0), (34, 16), (0, 0, 0), -1)
            cv2.putText(img, str(s + k), (2, 13), 0, 0.45, (0, 255, 255), 1)
            tiles.append(img)
        while len(tiles) % 8:
            tiles.append(np.zeros((120, 120, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[r : r + 8]) for r in range(0, len(tiles), 8)])
        cv2.imwrite(str(cache_dir("jersey") / f"sheet_{s // 40:02d}.jpg"), sheet)
    print(len(meta), "recortes")


if __name__ == "__main__":
    main()
