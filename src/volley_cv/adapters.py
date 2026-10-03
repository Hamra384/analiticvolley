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
    """YOLOv8 COCO, clase persona (SPIKE-001: YOLOv8m, imgsz 1280).

    La misma pasada detecta la pelota (clase 32); `YoloBallDetector.from_person_detector` la reutiliza
    sin volver a correr el modelo (SPEC-005 RF-1).
    """

    def __init__(
        self,
        weights: Path | None = None,
        imgsz: int = 1280,
        conf: float = 0.25,
        ball_conf: float = 0.05,
        model: Any | None = None,
    ) -> None:
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(weights))
        self.model = model
        self.imgsz, self.conf, self.ball_conf = imgsz, conf, ball_conf
        self._last: tuple[int, NDArray[np.float32]] | None = None  # (id del frame, pelotas de esa pasada)

    def detect(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        people, balls = self._run(frame)
        self._last = (id(frame), balls)
        return people

    def balls(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        if self._last is not None and self._last[0] == id(frame):
            return self._last[1]
        return self._run(frame)[1]

    def _run(self, frame: NDArray[np.uint8]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        r = self.model.predict(
            frame, classes=[0, 32], imgsz=self.imgsz, conf=min(self.conf, self.ball_conf), verbose=False
        )[0]
        b = r.boxes
        xyxy = np.asarray(b.xyxy.cpu().numpy(), dtype=np.float32).reshape(-1, 4)
        conf = np.asarray(b.conf.cpu().numpy(), dtype=np.float32).reshape(-1)
        cls = np.asarray(b.cls.cpu().numpy()).reshape(-1).astype(int)
        p = (cls == 0) & (conf >= self.conf)
        people = np.concatenate([xyxy[p], conf[p, None]], axis=1).reshape(-1, 5)
        k = (cls == 32) & (conf >= self.ball_conf)
        return people.astype(np.float32), _centers(xyxy[k], conf[k])


class YoloBallDetector:
    """Pelota (SPEC-005 RF-1): pasada COCO del detector de personas, o pesos propios de 1 clase."""

    def __init__(
        self, weights: Path | None = None, imgsz: int = 1280, conf: float = 0.05, model: Any | None = None
    ) -> None:
        self._shared: YoloPersonDetector | None = None
        if model is None and weights is not None:
            from ultralytics import YOLO

            model = YOLO(str(weights))
        self.model, self.imgsz, self.conf = model, imgsz, conf

    @classmethod
    def from_person_detector(cls, people: YoloPersonDetector) -> YoloBallDetector:
        out = cls()
        out._shared = people
        return out

    def detect(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        if self._shared is not None:
            return self._shared.balls(frame)
        assert self.model is not None
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        b = r.boxes
        xyxy = np.asarray(b.xyxy.cpu().numpy(), dtype=np.float32).reshape(-1, 4)
        return _centers(xyxy, np.asarray(b.conf.cpu().numpy(), dtype=np.float32).reshape(-1))


def _centers(xyxy: NDArray[np.float32], conf: NDArray[np.float32]) -> NDArray[np.float32]:
    """(x1, y1, x2, y2) + conf -> (cx, cy, ancho, alto, conf)."""
    out = np.stack(
        [
            (xyxy[:, 0] + xyxy[:, 2]) / 2,
            (xyxy[:, 1] + xyxy[:, 3]) / 2,
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1],
            conf,
        ],
        axis=1,
    )
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
