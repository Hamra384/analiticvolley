"""Pre-anotación de los clips de evaluación para corregir en CVAT (S2, Issue #5).

Por clip genera, en <data>/annotations/cvat/<clip>/:
  images.zip  frames JPEG 000001.jpg… (numeración 1-based, igual que MOT)
  mot.zip     gt/gt.txt + gt/labels.txt (formato MOT 1.1, importable en CVAT)

Jugadores: tracks de ByteTrack (SPIKE-001) filtrados a cajas de altura ≥ 7 % del frame y tracks ≥ 5 frames.
Pelota: track del enlazador de SPIKE-002 con el detector indicado (una caja chica centrada en la posición).
Es una propuesta para acelerar la corrección humana, no ground truth.

Uso: uv run --extra ml python -m tools.preannotate [--clips A1 ...] [--ball-detector B1|B2|B3]
"""

from __future__ import annotations

import argparse
import io
import zipfile

import cv2
import numpy as np

from spikes.ball.run_spike import classic_candidates, link, yolo_candidates
from spikes.common import ClipInfo, cache_dir, clips, data_dir, frames

LABELS = ["player", "ball"]  # class_id = índice + 1
PLAYER_MIN_H = 0.07
MIN_TRACK_LEN = 5
BALL_ID = 1000


def player_rows(info: ClipInfo) -> list[str]:
    tr = np.load(cache_dir("tracks") / f"{info.clip.id}_T1_bytetrack.npy")
    tr = tr[(tr[:, 5] - tr[:, 3]) >= PLAYER_MIN_H * info.height]
    ids, counts = np.unique(tr[:, 1], return_counts=True)
    keep = set(ids[counts >= MIN_TRACK_LEN].astype(int).tolist())
    rows = []
    for f, tid, x1, y1, x2, y2 in tr:
        if int(tid) in keep:
            rows.append(f"{int(f) + 1},{int(tid) + 1},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,1,1.0")
    return rows


def ball_rows(info: ClipInfo, detector: str) -> list[str]:
    if detector == "B3":
        cands, _ = classic_candidates(info)
    else:
        cands, _ = yolo_candidates(info, "yolov8m.pt" if detector == "B1" else "yolov8l.pt")
    track, _ = link(cands, info.width)
    r = 0.009 * info.height
    return [
        f"{i + 1},{BALL_ID},{x - r:.1f},{y - r:.1f},{2 * r:.1f},{2 * r:.1f},1,2,1.0"
        for i, p in enumerate(track)
        if p is not None
        for x, y in [p]
    ]


def export(info: ClipInfo, detector: str) -> None:
    out = data_dir() / "annotations" / "cvat" / info.clip.id
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / "images.zip", "w", zipfile.ZIP_STORED) as z:
        for i, frame in frames(info):
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                z.writestr(f"{i + 1:06d}.jpg", buf.tobytes())
    rows = player_rows(info) + ball_rows(info, detector)
    with zipfile.ZipFile(out / "mot.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("gt/gt.txt", "\n".join(rows) + "\n")
        z.writestr("gt/labels.txt", "\n".join(LABELS) + "\n")
    s = io.StringIO()
    print(f"{info.clip.id}: {len(rows)} cajas -> {out}", file=s)
    print(s.getvalue().strip(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--ball-detector", default="B1", choices=["B1", "B2", "B3"])
    args = ap.parse_args()
    for info in clips(args.clips):
        export(info, args.ball_detector)


if __name__ == "__main__":
    main()
