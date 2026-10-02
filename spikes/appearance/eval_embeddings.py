"""SPIKE-003 (S4.1): ¿el descriptor de apariencia distingue jugadores reales? ResNet18 vs OSNet vs color.

Pares positivos: mismo track de ByteTrack separado 10 frames (casi seguro la misma persona).
Pares negativos: dos tracks distintos en el mismo frame (seguro personas distintas); subgrupo "difícil":
torsos de color parecido (compañeros de equipo). Métrica: AUC = P(d_negativo > d_positivo).

Uso: uv run --extra ml python -m spikes.appearance.eval_embeddings [--clips A2 K2]
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

import cv2
import numpy as np

from spikes.common import cache_dir, clips, data_dir, frames
from volley_cv.team import torso_lab

GAP, N_PAIRS, MIN_H = 10, 400, 0.07


class ColorHist:
    name = "color_hist"

    def embed(self, frame: np.ndarray, boxes: list[tuple[float, ...]]) -> list[np.ndarray | None]:
        out: list[np.ndarray | None] = []
        for x1, y1, x2, y2 in boxes:
            c = frame[max(int(y1), 0) : int(y2), max(int(x1), 0) : int(x2)]
            if c.size == 0:
                out.append(None)
                continue
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256]).ravel()
            out.append(np.sqrt(h / max(h.sum(), 1)).astype(np.float32))
        return out


class OsNet:
    name = "osnet_x0_25_msmt17"

    def __init__(self) -> None:
        import torch
        from boxmot.reid import ReIDEncoderSpec, create_reid_encoder
        from boxmot.structures import Boxes, Detections, Frame

        self.torch, self.Boxes, self.Detections, self.Frame = torch, Boxes, Detections, Frame
        import hashlib

        weights = data_dir() / "weights" / "osnet_x0_25_msmt17.pt"
        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.enc = create_reid_encoder(
            ReIDEncoderSpec(backend="pytorch", artifact=str(weights), artifact_sha256=sha, device=dev)
        )

    def embed(self, frame: np.ndarray, boxes: list[tuple[float, ...]]) -> list[np.ndarray | None]:
        t = self.torch
        img = t.from_numpy(np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1)))
        f = self.Frame(image=img, sample_id="s")
        b = t.tensor(boxes, dtype=t.float32).reshape(-1, 4)
        d = self.Detections(
            geometry=self.Boxes(b),
            scores=t.ones(len(boxes)),
            class_ids=t.zeros(len(boxes), dtype=t.int64),
            sample_id="s",
        )
        feats = self.enc.encode([f], [d])[0].numpy()
        return [x.astype(np.float32) for x in feats]


def resnet() -> Any:
    from volley_cv.adapters import TorchvisionEmbedder

    e = TorchvisionEmbedder()
    e.name = "resnet18_imagenet"  # type: ignore[attr-defined]
    return e


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def auc(pos: list[float], neg: list[float]) -> float:
    p, n = np.array(pos), np.array(neg)
    return float((n[None, :] > p[:, None]).mean() + 0.5 * (n[None, :] == p[:, None]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=["A2", "K2"])
    args = ap.parse_args()
    rng = random.Random(0)
    embedders: list[Any] = [ColorHist(), resnet()]
    try:
        embedders.append(OsNet())
    except Exception as e:  # registrar y seguir: el spike documenta si OSNet no se pudo cargar
        print(json.dumps({"osnet_error": f"{type(e).__name__}: {e}"[:400]}), flush=True)
    results = []
    for info in clips(args.clips):
        tr = np.load(cache_dir("tracks") / f"{info.clip.id}_T1_bytetrack.npy")
        tr = tr[(tr[:, 5] - tr[:, 3]) >= MIN_H * info.height]
        by_frame: dict[int, list[np.ndarray]] = {}
        for row in tr:
            by_frame.setdefault(int(row[0]), []).append(row)
        key = {(int(r[0]), int(r[1])): r for r in tr}
        pos = [(f, t, f + GAP) for (f, t) in key if (f + GAP, t) in key]
        neg = [
            (f, a[1], b[1]) for f, rows in by_frame.items() for i, a in enumerate(rows) for b in rows[i + 1 :]
        ]
        pos = rng.sample(pos, min(N_PAIRS, len(pos)))
        neg = rng.sample(neg, min(4 * N_PAIRS, len(neg)))
        need: dict[int, set[int]] = {}
        for f, t, f2 in pos:
            need.setdefault(f, set()).add(t)
            need.setdefault(f2, set()).add(t)
        for f, a, b in neg:
            need.setdefault(f, set()).update({int(a), int(b)})
        feats: dict[str, dict[tuple[int, int], np.ndarray]] = {e.name: {} for e in embedders}
        colors: dict[tuple[int, int], np.ndarray] = {}
        for i, frame in frames(info):
            if i not in need:
                continue
            tids = sorted(need[i])
            boxes = [tuple(float(v) for v in key[(i, t)][2:6]) for t in tids]
            for t, b in zip(tids, boxes, strict=True):
                lab = torso_lab(frame, b)  # type: ignore[arg-type]
                if lab is not None:
                    colors[(i, t)] = lab
            for e in embedders:
                for t, v in zip(tids, e.embed(frame, boxes), strict=True):
                    if v is not None:
                        feats[e.name][(i, t)] = unit(np.asarray(v, dtype=np.float32))
        for e in embedders:
            fe = feats[e.name]
            d_pos = [
                1 - float(fe[(f, t)] @ fe[(f2, t)]) for f, t, f2 in pos if (f, t) in fe and (f2, t) in fe
            ]
            negs = [(f, int(a), int(b)) for f, a, b in neg if (f, int(a)) in fe and (f, int(b)) in fe]
            d_neg = [1 - float(fe[(f, a)] @ fe[(f, b)]) for f, a, b in negs]
            hard = [
                1 - float(fe[(f, a)] @ fe[(f, b)])
                for f, a, b in negs
                if (f, a) in colors
                and (f, b) in colors
                and np.linalg.norm(colors[(f, a)] - colors[(f, b)]) < 20
            ]
            r = {
                "clip": info.clip.id,
                "embedder": e.name,
                "n_pos": len(d_pos),
                "n_neg": len(d_neg),
                "n_hard": len(hard),
                "auc_all": round(auc(d_pos, d_neg), 3),
                "auc_hard_same_color": round(auc(d_pos, hard), 3) if hard else None,
                "pos_median": round(float(np.median(d_pos)), 3),
                "pos_p90": round(float(np.percentile(d_pos, 90)), 3),
                "hard_median": round(float(np.median(hard)), 3) if hard else None,
                "hard_p10": round(float(np.percentile(hard, 10)), 3) if hard else None,
            }
            print(json.dumps(r), flush=True)
            results.append(r)
    (cache_dir("reports") / "spike003_results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
