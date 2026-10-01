# legacy/ — scripts originales (congelados)

Referencia histórica del prototipo previo al MVP 1. **No se mantienen ni se importan** desde `src/`.
Están excluidos de pytest, ruff y mypy. Contienen rutas absolutas hardcodeadas (deuda conocida; el código nuevo
resuelve rutas vía `volley_cv.config`).

| Script | Original | Qué hace |
|---|---|---|
| `run_tracking_ocr.py` | `test_ocr.py` | Loop interactivo: YOLOv8n + ByteTrack, OCR asíncrono (easyocr), pelota COCO cada 5 frames |
| `run_deteccion.py` | `test_deteccion.py` | Visualización de tracking YOLOv8m |
| `diag_ocr.py` | — | Diagnóstico de OCR sobre 600 frames |
| `recopilar_datos.py` | — | Auto-etiquetado de crops de dorsales con easyocr |
| `entrenar_dorsal.py` | — | Entrena MobileNetV3 → `models/dorsal_classifier.onnx` |
| `train_svhn.py` | — | Entrena CNN de dígitos → `models/digit_classifier.onnx` |

Se renombraron `test_*.py` → `run_*.py` porque no son tests y pytest los habría ejecutado.
Para correrlos: `cd legacy && python run_tracking_ocr.py` (usan `bytetrack_custom.yaml` relativo).

## Defectos conocidos (auditoría 2026-10-01; no se corrigen acá, se resuelven en el diseño nuevo)

1. `run_tracking_ocr.py:350` — la herencia de número exige `votos`, que no existen para números confirmados por
   alta confianza (≥0.88): justo los más confiables nunca se heredan.
2. `run_tracking_ocr.py:178` — unicidad de número global, no por equipo.
3. Herencia de identidad por una sola señal (distancia < 180 px).
4. Equipo inferido por lado de cancha (cerca/lejos).
5. Pelota sin tracking temporal; posición congelada entre detecciones.
6. `entrenar_dorsal.py` — `random_split` mezcla frames consecutivos del mismo jugador entre train/val (accuracy
   inflada). Las etiquetas automáticas tienen ruido: `models/dorsal_classes.json` contiene `"0"`, `"03"`, `"06"`.
