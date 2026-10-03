"""Fine-tuning del detector de pelota (SPEC-005 RF-7) sobre `datasets/ball_v1` (pseudo-etiquetas).

YOLOv8s preentrenado en COCO, 1 clase, imgsz 1280 (GTX 1660 Super 6 GB: batch chico). Los pesos quedan en
`<datos>/models/ball_v1/` (derivan de transmisiones con derechos de terceros: nunca en el repo).

Uso: uv run --extra ml python -m tools.train_ball [--epochs 40] [--batch 4]
"""

from __future__ import annotations

import argparse
import shutil
import sys

from volley_cv.config import load_settings


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--base", default="yolov8s.pt")
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    data = load_settings().data_dir
    base = data / args.base
    model = YOLO(str(base) if base.is_file() else args.base)  # si no está, ultralytics la descarga (gratis)
    out = data / "models"
    model.train(
        data=str(data / "datasets" / "ball_v1" / "data.yaml"),
        imgsz=1280,
        epochs=args.epochs,
        batch=args.batch,
        patience=10,
        project=str(out),
        name="ball_v1",
        exist_ok=True,
        workers=2,
        seed=0,
        deterministic=False,  # con True, torch 2.5 + CUDA falla en torch.unique (overflow)
        # la pelota es chica y borrosa: sin recortes agresivos; blur/HSV moderados
        mosaic=0.5,
        scale=0.3,
        degrees=0.0,
        fliplr=0.5,
        hsv_v=0.3,
        verbose=False,
    )
    best = out / "ball_v1" / "weights" / "best.pt"
    shutil.copy2(best, out / "ball_v1.pt")
    print(f"pesos: {out / 'ball_v1.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
