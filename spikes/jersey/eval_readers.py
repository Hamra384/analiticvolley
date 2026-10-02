"""SPIKE-004: compara lectores de número de camiseta sobre el set etiquetado a mano.

Precisión = lecturas correctas / lecturas emitidas (inventar un número donde no hay cuenta como error).
Cobertura = lecturas correctas / recortes con número visible. Se reporta por umbral de confianza y el mejor
punto con precisión >= 0,95 (criterio fijado antes de medir: un número equivocado es peor que ninguno).

Uso: uv run --extra ml python -m spikes.jersey.eval_readers
"""

from __future__ import annotations

import json
from collections.abc import Callable

import cv2
import numpy as np

from spikes.common import PROJECT_ROOT, cache_dir

Reader = Callable[[np.ndarray], tuple[int | None, float]]


def _best_digits(results: list[tuple[object, str, float]]) -> tuple[int | None, float]:
    best: tuple[int | None, float] = (None, 0.0)
    for _, text, conf in results:
        t = text.strip()
        if t.isdigit() and 1 <= len(t) <= 2 and float(conf) > best[1]:
            best = (int(t), float(conf))
    return best


def easyocr_reader(variant: str) -> Reader:
    import easyocr

    ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

    def read(crop: np.ndarray) -> tuple[int | None, float]:
        scale = 160 / max(crop.shape[0], 1)
        img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs = [gray] if variant == "raw" else [clahe.apply(gray), 255 - clahe.apply(gray)]
        best: tuple[int | None, float] = (None, 0.0)
        for im in imgs:
            res = ocr.readtext(im, allowlist="0123456789", text_threshold=0.4, low_text=0.3)
            cand = _best_digits(res)
            if cand[1] > best[1]:
                best = cand
        return best

    return read


def legacy_classifier() -> Reader:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(PROJECT_ROOT / "models" / "dorsal_classifier.onnx"))
    classes = json.loads((PROJECT_ROOT / "models" / "dorsal_classes.json").read_text(encoding="utf-8"))
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def read(crop: np.ndarray) -> tuple[int | None, float]:
        x = cv2.resize(crop, (96, 96))[:, :, ::-1] / 255.0
        x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        k = int(p.argmax())
        return int(classes[k]), float(p[k])

    return read


def evaluate(name: str, reader: Reader, items: list[tuple[np.ndarray, int | None]]) -> dict[str, object]:
    preds = [reader(img) for img, _ in items]
    n_num = sum(gt is not None for _, gt in items)
    curve = []
    for t in [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]:
        emitted = [(p, gt) for (p, c), (_, gt) in zip(preds, items, strict=True) if p is not None and c >= t]
        correct = sum(p == gt for p, gt in emitted)
        prec = correct / len(emitted) if emitted else None
        curve.append({"t": t, "emitted": len(emitted), "precision": prec, "recall": correct / n_num})
    ok = [c for c in curve if c["precision"] is not None and c["precision"] >= 0.95]
    best = max(ok, key=lambda c: c["recall"]) if ok else None
    return {"reader": name, "best_at_p95": best, "curve": curve}


def main() -> None:
    labels = json.loads((cache_dir("jersey") / "labels.json").read_text(encoding="utf-8"))
    items = []
    for cid, lab in labels.items():
        if lab == "?":
            continue
        img = cv2.imread(str(cache_dir("jersey", "crops") / f"{cid}.png"))
        items.append((img, lab))
    results = []
    for name, factory in (
        ("easyocr_raw", lambda: easyocr_reader("raw")),
        ("easyocr_clahe_2pol", lambda: easyocr_reader("clahe")),
        ("legacy_mobilenet", legacy_classifier),
    ):
        r = evaluate(name, factory(), items)
        results.append(r)
        b = r["best_at_p95"]
        print(json.dumps({"reader": name, "n": len(items), "best_at_p95": b}), flush=True)
        for c in r["curve"][::3]:  # type: ignore[index]
            print("   ", c, flush=True)
    (cache_dir("reports") / "spike004_results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
