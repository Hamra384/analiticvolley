# SPIKE-002: ¿Cómo detectar la pelota con suficiente cobertura para trackearla?

- **Issue:** #5
- **Time-box:** 1 sesión de trabajo
- **Criterio de finalización:** métricas proxy de 3 detectores sobre los 10 clips, verificación visual de una muestra,
  y recomendación del camino a seguir para el `BallDetector`.

## Pregunta e hipótesis

La pelota mide ~10–20 px (720p/1080p), se mueve rápido y tiene motion blur. En la auditoría, YOLOv8m COCO la
detectó en ~15–45 % de las muestras por ventana. **Hipótesis:** un detector genérico COCO no alcanza la cobertura
necesaria (objetivo del MVP: recall ≥ 0,80), y un detector clásico por movimiento + color, filtrado
temporalmente, la mejora en este tipo de video (fondo con cámara casi fija entre paneos, pelota amarilla/azul).

## Criterios de decisión (fijados antes de medir)

Todos los detectores alimentan el **mismo enlazador temporal simple** (predicción de velocidad constante + gating
de distancia), que produce un único track de pelota por clip.

| Métrica | Definición | Mejor |
|---|---|---|
| **P1 · Cobertura** (primaria) | Frames con posición enlazada a una detección real (no predicha) / frames del clip | mayor |
| **P2 · Plausibilidad** | Detecciones crudas que terminan en el track principal / detecciones crudas | mayor |
| **P3 · Saltos imposibles** | Pasos del track con desplazamiento > 5 % del ancho del frame en un frame | menor |
| **P4 · Precisión visual** | Sobre 12 frames al azar por detector con posición reportada: proporción en la que la marca cae sobre la pelota (verificación visual manual) | mayor |
| **P5 · FPS** | Solo detección, GTX 1660 Super | se reporta |

Aclaración (antes de medir): como el detector clásico produce varios candidatos por frame, P2 se calcula por
frame: frames en los que el track usó un candidato / frames con al menos un candidato.

**Regla de decisión:** se elige el de mayor P1 entre los que tengan P2 ≥ 0,6 y P4 ≥ 0,8. Si ninguno llega a
P1 ≥ 0,6, la recomendación es entrenar un detector específico (fine-tuning) con los datos que se anoten en CVAT.

Limitación conocida: sin ground truth, P1 mide cobertura del track, no recall real. Se mide recall real en S6.

## Alternativas evaluadas

| ID | Detector |
|---|---|
| B1 | YOLOv8m COCO clase 32 ("sports ball"), imgsz 1280, conf ≥ 0,05 |
| B2 | YOLOv8l COCO clase 32, imgsz 1280, conf ≥ 0,05 |
| B3 | Clásico: diferencia de 3 frames + máscara de color de la pelota + filtro de tamaño y circularidad |

Descartado de antemano: datasets públicos de Roboflow Universe (requieren cuenta y API key; queda como opción si
P1 no alcanza, previa revisión de licencia y con tu cuenta).

## Método
`spikes/ball/run_spike.py`. Hardware: GTX 1660 Super 6 GB.

## Resultados (2026-10-02)

Métricas automáticas, 10 clips (fuente: `<data>/cache/reports/spike002_results.json`):

| Detector | P1 cobertura mediana (mín) | P2 plausibilidad mediana | P3 saltos (suma) | % frames con candidato | P5 FPS |
|---|---|---|---|---|---|
| B1 YOLOv8m COCO | 0,359 (0,268) | 0,727 | 0 | 0,53 | 16,7 |
| B2 YOLOv8l COCO | 0,359 (0,273) | 0,756 | 0 | 0,50 | 10,2 |
| B3 clásico (movimiento + color) | 0,732 (0,538) | 0,744 | 25 | 0,99 | 54,7 |

Verificación visual P4 (12 recortes al azar por clip, marca sobre la pelota sí/no):

| Detector | Clips revisados | Aciertos | P4 |
|---|---|---|---|
| B1 | A2, K1 | 22 / 24 | **0,92** |
| B3 | A2, A5, K2, K3 | ~7 / 48 | **~0,15** |
| B2 | — | no revisado (empata en P1 con B1 y es 1,6× más lento) | — |

Observaciones (hechos):
- La cobertura alta de B3 es **falsa**: en K2/K3/A5 el track sigue carteles LED animados (azul/amarillo) y
  bordes de publicidad, no la pelota. Solo en A2 (fondo sin LED) acierta la mitad de las veces.
- B1/B2 son precisos (cuando dicen "pelota", es la pelota) pero pierden la pelota en ~2/3 de los frames
  (pelota chica, motion blur, pelota sobre el público).

## Conclusión y recomendación

- B3 descartado (P4 < 0,8). B1 cumple P2 y P4; B1 y B2 empatan en P1 → **B1** por velocidad.
- **Ningún detector llega a P1 ≥ 0,6** → según la regla, la recomendación es **entrenar un detector específico de
  pelota** (S5).
- Plan propuesto para el detector específico (sin costo, sin cuentas externas):
  1. Pseudo-etiquetas en segmentos de los dos videos **fuera** de los clips de evaluación (margen ±60 s, para no
     filtrar datos de evaluación al entrenamiento): detecciones B1 de alta confianza + enlazador + interpolación
     corta, filtradas por consistencia de trayectoria.
  2. Fine-tuning de YOLOv8 (1 clase, imgsz 1280) sobre esas pseudo-etiquetas; validación contra las anotaciones
     humanas de los clips de evaluación.
  3. Alternativa si (2) no alcanza: dataset público de vóley (Roboflow Universe, requiere cuenta gratuita del
     usuario), previa revisión de licencia en `docs/data/DATASETS.md`.
- Mientras tanto, la pre-anotación de la pelota usa B1 (precisa; el humano completa los frames faltantes).
- El `BallTracker` (Kalman + estados DETECTED/PREDICTED/LOST) es necesario igual: con cualquier detector va a haber
  huecos de varios frames.
