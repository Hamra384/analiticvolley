"""Dataset de pelota con pseudo-etiquetas (SPEC-005 RF-7).

Ventanas cortas repartidas por los dos videos, **fuera** de ±60 s de cualquier clip de evaluación. En cada
ventana corre YOLOv8m COCO (clase 32) y el BallTracker; una trayectoria (detecciones con huecos de ≤ 3 frames)
se acepta si tiene ≥ 5 detecciones y alguna con confianza ≥ 0,25. Se etiquetan sus detecciones y se interpolan
los huecos.
Solo se guardan frames con etiqueta (un frame sin etiqueta puede tener una pelota que el detector no vio).

Las imágenes salen de transmisiones con derechos de terceros: van al directorio de datos, nunca al repo.

Uso: uv run --extra ml python -m tools.ball_dataset   ->  <datos>/datasets/ball_v1/{images,labels}/{train,val}
"""

from __future__ import annotations

import itertools
import json
import sys

import cv2
import numpy as np

from volley_cv.ball import BallTracker
from volley_cv.config import PROJECT_ROOT, ClipCatalog, load_clips, load_settings

MIN_DETS, MAX_GAP, ANCHOR_CONF, CAND_CONF = 5, 3, 0.25, 0.1
SAVE_EVERY = 2  # de cada trayectoria se guarda 1 de cada 2 frames etiquetados (menos redundancia)
VAL_EVERY = 7  # 1 de cada 7 ventanas va a validación


def training_windows(
    catalog: ClipCatalog,
    durations: dict[str, float],
    margin_s: float = 60.0,
    window_s: float = 2.0,
    every_s: float = 20.0,
) -> list[tuple[str, float, float]]:
    """(video, inicio, fin): ventanas de `window_s` cada `every_s`, a `margin_s` o más de los clips de eval."""
    out = []
    for video, dur in sorted(durations.items()):
        banned = [(c.start_s - margin_s, c.end_s + margin_s) for c in catalog.clips if c.video == video]
        t = 5.0
        while t + window_s <= dur:
            end = t + window_s
            if all(end <= a or t >= b for a, b in banned):
                out.append((video, t, end))
            t += every_s
    return out


def trajectories(dets: list[tuple[float, float, float, float, float] | None]) -> list[list[int]]:
    """Índices de frames de cada trayectoria aceptada (detecciones con huecos de ≤ MAX_GAP frames)."""
    runs: list[list[int]] = []
    cur: list[int] = []
    for i, d in enumerate(dets):
        if d is None:
            continue
        if cur and i - cur[-1] > MAX_GAP + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= MIN_DETS and max(dets[i][4] for i in r) >= ANCHOR_CONF]  # type: ignore[index]


def labels_for(
    dets: list[tuple[float, float, float, float, float] | None],
) -> dict[int, tuple[float, float, float, float]]:
    """Etiquetas (cx, cy, w, h) por frame: detecciones de trayectorias aceptadas + huecos interpolados."""
    out: dict[int, tuple[float, float, float, float]] = {}
    for run in trajectories(dets):
        for a, b in itertools.pairwise(run):
            da, db = dets[a], dets[b]
            assert da is not None and db is not None
            out[a] = da[:4]
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                out[k] = tuple(float(x) for x in np.array(da[:4]) * (1 - t) + np.array(db[:4]) * t)  # type: ignore[assignment]
        last = dets[run[-1]]
        assert last is not None
        out[run[-1]] = last[:4]
    return out


def main(argv: list[str]) -> int:
    from ultralytics import YOLO

    data = load_settings().data_dir
    catalog = load_clips(PROJECT_ROOT / "configs" / "eval" / "clips.yaml")
    durations = {}
    for vid, v in catalog.videos.items():
        cap = cv2.VideoCapture(str(data / v.file))
        durations[vid] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    wins = training_windows(catalog, durations)
    root = data / "datasets" / "ball_v1"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    model = YOLO(str(data / "yolov8m.pt"))
    stats = {"windows": len(wins), "images": 0, "train": 0, "val": 0}
    for w, (vid, start, end) in enumerate(wins):
        cap = cv2.VideoCapture(str(data / catalog.videos[vid].file))
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        frames, dets = [], []
        tracker = BallTracker()
        for _ in range(round((end - start) * cap.get(cv2.CAP_PROP_FPS))):
            ok, frame = cap.read()
            if not ok:
                break
            r = model.predict(frame, classes=[32], imgsz=1280, conf=CAND_CONF, verbose=False)[0]
            xyxy = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
            conf = r.boxes.conf.cpu().numpy().reshape(-1)
            cands = np.stack(
                [
                    (xyxy[:, 0] + xyxy[:, 2]) / 2,
                    (xyxy[:, 1] + xyxy[:, 3]) / 2,
                    xyxy[:, 2] - xyxy[:, 0],
                    xyxy[:, 3] - xyxy[:, 1],
                    conf,
                ],
                axis=1,
            ).reshape(-1, 5)
            st = tracker.update(cands.astype(np.float32))
            chosen = None
            if st.state in ("DETECTED", "TRACKED", "REACQUIRED") and st.position is not None and len(cands):
                k = int(np.argmin(np.hypot(cands[:, 0] - st.position[0], cands[:, 1] - st.position[1])))
                chosen = tuple(float(v) for v in cands[k])
            frames.append(frame)
            dets.append(chosen)
        cap.release()
        split = "val" if w % VAL_EVERY == 0 else "train"
        labs = labels_for(dets)  # type: ignore[arg-type]
        for n, (i, (cx, cy, bw, bh)) in enumerate(sorted(labs.items())):
            if n % SAVE_EVERY:
                continue
            h, wd = frames[i].shape[:2]
            name = f"{vid}_{int(start):05d}_{i:03d}"
            cv2.imwrite(
                str(root / "images" / split / f"{name}.jpg"), frames[i], [cv2.IMWRITE_JPEG_QUALITY, 92]
            )
            (root / "labels" / split / f"{name}.txt").write_text(
                f"0 {cx / wd:.6f} {cy / h:.6f} {max(bw, 4) / wd:.6f} {max(bh, 4) / h:.6f}\n", encoding="utf-8"
            )
            stats["images"] += 1
            stats[split] += 1
        print(f"{w + 1}/{len(wins)} {vid} {start:.0f}s: {len(labs)} etiquetas", flush=True)
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n",
        encoding="utf-8",
    )
    (root / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
