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

## Datasets de terceros

| Nombre | Fuente | Licencia | Uso previsto | Componente | Estado |
|---|---|---|---|---|---|
| SVHN | torchvision | Uso no comercial (Stanford) | Entrenamiento de `digit_classifier.onnx` (legacy) | JerseyReader (a evaluar en S5) | Legacy |
| *(pendiente)* dataset público de pelota de vóley | — | Revisar antes de usar | Fine-tuning del detector de pelota | BallDetector | Se evalúa en el spike de S2 |

## Anotaciones

Ground truth de los 10 clips de `configs/eval/clips.yaml`: se pre-anotan automáticamente y el usuario las corrige
en CVAT (S2). Se guardan en el directorio de datos (`annotations/`), no en el repo.
