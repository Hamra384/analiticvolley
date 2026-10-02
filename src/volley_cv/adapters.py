"""Implementaciones reales de los componentes con modelos (extra `ml`; importan torch/ultralytics/boxmot de
forma diferida). No se testean en CI (requieren GPU y pesos); su uso real se evidencia en el informe de S4."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

Box = tuple[float, float, float, float]


def _device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class YoloPersonDetector:
    """YOLOv8 COCO, clase persona (SPIKE-001: YOLOv8m, imgsz 1280)."""

    def __init__(self, weights: Path, imgsz: int = 1280, conf: float = 0.25) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz, self.conf = imgsz, conf

    def detect(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        r = self.model.predict(frame, classes=[0], imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        b = r.boxes
        out = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], axis=1)
        return np.asarray(out, dtype=np.float32).reshape(-1, 5)


class ByteTrackTracker:
    """ByteTrack de boxmot (SPIKE-001)."""

    def __init__(self, frame_rate: int = 30) -> None:
        self.frame_rate = frame_rate
        self.reset()

    def reset(self) -> None:
        from boxmot import ByteTrack

        self._t: Any = ByteTrack(min_conf=0.1, track_thresh=0.4, track_buffer=30, frame_rate=self.frame_rate)

    def update(self, detections: NDArray[np.float32], frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        dets = np.zeros((len(detections), 6), dtype=np.float32)
        dets[:, :5] = detections  # clase 0 = persona
        tracks = self._t.update(dets, frame)
        if tracks is None or len(tracks) == 0:
            return np.empty((0, 6), dtype=np.float32)
        t = np.asarray(tracks, dtype=np.float32)
        return np.stack([t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5]], axis=1)


class TorchvisionEmbedder:
    """Descriptor de apariencia: ResNet18 ImageNet sobre el recorte 256x128 (pooling global, 512-d).

    Decisión S4 (SPEC-002 RF-4): reemplaza a OSNet x0.25 hasta resolver la API de Re-ID de boxmot 25; se
    compara en S6 con datos anotados.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self) -> None:
        import torch
        import torchvision

        self.torch = torch
        self.device = _device()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(self.device)

    def embed(self, frame: NDArray[np.uint8], boxes: Sequence[Box]) -> list[NDArray[np.float32] | None]:
        crops, idx = [], []
        h, w = frame.shape[:2]
        for k, (x1, y1, x2, y2) in enumerate(boxes):
            crop = frame[max(int(y1), 0) : min(int(y2), h), max(int(x1), 0) : min(int(x2), w)]
            if crop.size == 0:
                continue
            rgb = np.asarray(cv2.resize(crop, (128, 256)), dtype=np.float32)[:, :, ::-1] / 255.0
            crops.append((rgb - self.MEAN) / self.STD)
            idx.append(k)
        out: list[NDArray[np.float32] | None] = [None] * len(boxes)
        if not crops:
            return out
        batch = self.torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2)).to(self.device)
        with self.torch.no_grad():
            feats = self.net(batch).cpu().numpy().astype(np.float32)
        for k, f in zip(idx, feats, strict=True):
            out[k] = f
        return out
