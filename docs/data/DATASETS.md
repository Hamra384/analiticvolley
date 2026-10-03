# Inventario de datos

Regla: **ningún video ni dataset se versiona** (repo público). Viven en el directorio de datos
(`VOLLEY_DATA_DIR` / `configs/local.yaml`). Controles: `.gitignore`, hook `guard_bash`, `scripts/check_no_media.py` en CI.

Antes de incorporar una fuente nueva se agrega una fila con fuente, licencia, uso permitido y componente.

## Videos

| ID | Archivo (relativo a datos) | Fuente | Licencia / derechos | Uso | Componente | Estado |
|---|---|---|---|---|---|---|
| `jpn_arg_2026` | `videos/external/jpn_arg_2026.mp4` (1080p, 50 min, 910 MB) | YouTube `oFMjaZ_z3X8`, canal "Technical Volleyball" | Licencia estándar de YouTube; aparenta re-subida de transmisión de TV por un tercero. Descarga autorizada por el usuario (2026-10-01) conociendo que contraviene los ToS de YouTube | Solo evaluación/desarrollo local. No redistribuir | Evaluación (clips A1–A5) | Activo |
| `jpn_kor_2026` | `videos/external/jpn_kor_2026.mp4` (720p, 47 min, 632 MB) | YouTube `dgU-3rkwf4Q`, mismo canal | Ídem | Ídem | Evaluación (clips K1–K5) | Activo |
| `partido` | `videos/partido.mp4` (360p, 89 min) | Propio del usuario (transmisión UBA Vóley) | Sin verificar | Desarrollo | — | Excluido de evaluación (resolución insuficiente) |
| `partido2` | `videos/partido2.mp4` (360p, 49 min) | Propio del usuario (VNL FRA–BRA) | Sin verificar | Desarrollo | — | Excluido de evaluación (resolución insuficiente) |

### Material de referencia (no evaluación)

| ID | Ubicación (relativa a datos) | Fuente | Licencia / derechos | Uso | Estado |
|---|---|---|---|---|---|
| `datavolley4_na7ki` | `videos/tutorials/datavolley4_na7ki/` (10 videos, ~80 min, + `.info.json` y descripción) | Lista de YouTube "DATAVOLLEY 4 - TUTORIAL" (`PLMgeGsmCIcpY0nVwTG-qTMlIOZOKqhbtl`), canal Na7ki Tayra | Licencia estándar de YouTube; derechos del creador. Descarga pedida por el usuario (2026-10-02) conociendo que contraviene los ToS de YouTube | Solo estudio local: entender la codificación de acciones del juego (zonas de saque, recepción) para etapas posteriores al MVP 1. No redistribuir | Activo |

## Datasets de terceros

| Nombre | Fuente | Licencia | Uso previsto | Componente | Estado |
|---|---|---|---|---|---|
| SVHN | torchvision | Uso no comercial (Stanford) | Entrenamiento de `digit_classifier.onnx` (legacy) | JerseyReader (a evaluar en S5) | Legacy |
| *(pendiente)* dataset público de pelota de vóley | — | Revisar antes de usar | Fine-tuning del detector de pelota | BallDetector | Se evalúa en el spike de S2 |

## Modelos preentrenados y librerías de terceros (S2)

| Recurso | Fuente | Licencia | Uso | Dónde vive |
|---|---|---|---|---|
| YOLOv8m / YOLOv8l (COCO) | Ultralytics | AGPL-3.0 (pesos y librería); anotaciones COCO CC BY 4.0 | Detección de personas y pelota (spikes, pre-anotación) | `<data>/yolov8*.pt` |
| boxmot 25.0 | PyPI (mikel-brostrom/boxmot) | AGPL-3.0 | Trackers del SPIKE-001 | extra `ml` |
| OSNet x0.25 (MSMT17) | boxmot (descarga automática) | Pesos entrenados sobre MSMT17, dataset con licencia de **uso no comercial para investigación** | Re-ID en SPIKE-001 (T3/T4; ninguno fue elegido) | `<data>/weights/` |
| CVAT 2.77.1 | github.com/cvat-ai/cvat | MIT | Herramienta de anotación local (Docker) | `D:\AIVolley\cvat` (fuera del repo) |

Nota: el proyecto ya es AGPL-compatible por depender de Ultralytics; el repo es público. Si en algún momento se
quisiera uso comercial, hay que revisar Ultralytics/boxmot (licencia comercial) y no usar pesos MSMT17.

## Datos y modelos derivados (S5b)

| Artefacto | Ubicación (directorio de datos) | Origen | En el repo |
|---|---|---|---|
| `datasets/ball_v1` (1738 frames con pseudo-etiqueta de pelota) | fuera del repo | frames de `jpn_arg_2026` y `jpn_kor_2026` a más de 60 s de cualquier clip de evaluación; etiquetas automáticas de YOLOv8m COCO + BallTracker (`tools/ball_dataset.py`) | **no** (derechos de terceros) |
| `models/ball_v1.pt` (YOLOv8s, 1 clase, detector de pelota por defecto) | fuera del repo | fine-tuning de YOLOv8s COCO (AGPL-3.0, Ultralytics) sobre `ball_v1` (`tools/train_ball.py`, 36 épocas) | **no** (deriva de las transmisiones) |

Para reproducir: `uv run --extra ml python -m tools.ball_dataset` y luego `python -m tools.train_ball --batch 2`.

## Anotaciones

Ground truth de los 10 clips de `configs/eval/clips.yaml`: se pre-anotan automáticamente y el usuario las corrige
en CVAT (S2). Se guardan en el directorio de datos (`annotations/`), no en el repo.
