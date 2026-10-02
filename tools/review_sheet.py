"""Hoja de revisión visual de una corrida (SPEC-002 AC-12, paso obligatorio antes de cerrar un sprint).

Por identidad: 8 recortes equiespaciados del video de debug (frame relativo y estado), más un resumen de
frames, huecos y re-identificaciones. Sirve para ver de un vistazo si una identidad mezcla personas
(oficiales, compañeros) o si un jugador aparece repartido en varias identidades.

Las imágenes salen de transmisiones con derechos de terceros: se guardan en el directorio de datos, nunca
en el repo público.

Uso: uv run python -m tools.review_sheet <clip>   ->  <datos>/outputs/<clip>/review_ids.jpg + review.json
"""

from __future__ import annotations

import itertools
import json
import sys

import cv2
import numpy as np

from volley_cv.config import load_settings

N_CROPS, W, H = 8, 70, 150


def main(argv: list[str]) -> int:
    clip = argv[0]
    run = load_settings().data_dir / "outputs" / clip
    rows = [json.loads(x) for x in (run / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows:
        print("sin frames")
        return 1
    first = rows[0]["frame"]
    per: dict[str, list[tuple[int, list[float], str]]] = {}
    for r in rows:
        for p in r["players"]:
            per.setdefault(p["player_id"], []).append((r["frame"] - first, p["bbox"], p["state"]))
    summary = []
    cap = cv2.VideoCapture(str(run / "debug.mp4"))
    sheet = []
    for pid, obs in sorted(per.items()):
        fs = [f for f, _, _ in obs]
        gaps = sum(1 for a, b in itertools.pairwise(fs) if b - a > 1)
        reids = sum(1 for _, _, s in obs if s == "REIDENTIFIED")
        summary.append(
            {
                "player_id": pid,
                "frames": len(fs),
                "first": fs[0],
                "last": fs[-1],
                "gaps": gaps,
                "reidentified": reids,
            }
        )
        tiles = []
        for f, b, st in [obs[k] for k in np.linspace(0, len(obs) - 1, N_CROPS).astype(int)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            x1, y1, x2, y2 = (int(v) for v in b)
            crop = img[max(y1 - 20, 0) : y2 + 5, max(x1 - 5, 0) : x2 + 5] if ok else None
            tile = (
                cv2.resize(crop, (W, H)) if crop is not None and crop.size else np.zeros((H, W, 3), np.uint8)
            )
            tag = "" if st == "TRACKED" else " " + st[:3]
            cv2.putText(tile, f"{f}{tag}", (2, 12), 0, 0.38, (0, 255, 255), 1)
            tiles.append(tile)
        label = np.zeros((H, 64, 3), np.uint8)
        cv2.putText(label, pid[5] + pid[-2:], (4, 70), 0, 0.6, (255, 255, 255), 2)
        cv2.putText(label, f"{len(fs)}f", (4, 95), 0, 0.45, (200, 200, 200), 1)
        sheet.append(np.hstack([label, *tiles]))
    cap.release()
    cv2.imwrite(str(run / "review_ids.jpg"), np.vstack(sheet))
    (run / "review.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{len(summary)} identidades -> {run / 'review_ids.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
