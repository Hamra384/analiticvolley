# SPIKE-001: ¿Qué tracker de corto plazo usar como base del pipeline?

- **Issue:** #5
- **Time-box:** 1 sesión de trabajo (detección cacheada + 4 trackers × 10 clips)
- **Criterio de finalización:** tabla de métricas proxy para los 4 trackers sobre los 10 clips, inspección visual de
  2 clips con superposición, y recomendación (confirmar o cambiar ADR-0002).

## Pregunta e hipótesis

ADR-0002 eligió BoT-SORT por su compensación de movimiento de cámara (CMC), dado que la cámara hace paneo.
**Hipótesis:** BoT-SORT con CMC produce menos intercambios de identidad que ByteTrack, y agregar embeddings de
apariencia (Re-ID) reduce aún más los intercambios entre jugadores de distinto color.

Recordatorio de diseño (ADR-0002): el tracker solo genera *tracklets* de corto plazo; unirlos es tarea del
`IdentityManager`. Por eso **un tracklet cortado es aceptable; un tracklet que salta de un jugador a otro no.**

## Criterios de decisión (fijados antes de medir)

Sin ground truth todavía, se usan métricas proxy calculadas igual para todos los trackers, sobre las **mismas
detecciones** (YOLOv8m COCO "person", imgsz 1280, conf ≥ 0,10) y solo para cajas de "jugador" (altura ≥ 7 % del
alto del frame, para excluir público).

| Métrica | Definición | Mejor |
|---|---|---|
| **M1 · Cambios de color** (primaria) | Tracklets cuyo color de torso cambia de grupo de forma sostenida (≥ 15 frames antes y después), con grupos por k-means (k=3) sobre el color de torso del clip. Proxy de intercambio de ID entre jugadores de distinto equipo o líbero | menor |
| **M2 · Fragmentación** | (IDs distintos) / (mediana de jugadores simultáneos). 1,0 = un ID por persona | menor |
| **M3 · Vida media** | Mediana de la duración de los tracklets / duración del clip | mayor |
| **M4 · FPS del tracker** | Solo el paso de tracking (sin detección), GTX 1660 Super | ≥ 10 requerido |

**Regla de decisión:** se descarta todo tracker con M4 < 10. Gana el menor M1 total. Si dos trackers quedan a
≤ 1 cambio de color de diferencia, desempata el menor M2. Si el ganador no es BoT-SORT, se actualiza ADR-0002.

Limitación conocida: M1 no detecta intercambios entre dos jugadores del **mismo** color. Se complementa con
inspección visual de los clips de superposición (A2, K2) y, en S6, con IDF1 sobre ground truth.

### Enmienda (2026-10-01, antes de comparar trackers)

Al validar M1 a ojo sobre A2/T3 (tiras de recortes por track) se observó:
- **Falso positivo:** un oficial (track #8), idéntico todo el clip, fue marcado como cambio de color.
- **Falsos negativos:** intercambios entre compañeros del mismo color (35 → 28 → 24) no cambian el color.

M1 no es confiable como métrica primaria. Se agrega, **antes de comparar trackers**, una auditoría visual:

| Métrica | Definición | Mejor |
|---|---|---|
| **M5 · Intercambios por track** (nueva primaria) | En A2 y K2, los 8 tracks de jugador más largos por tracker; tira de 10 recortes equiespaciados; se cuenta cuántas personas distintas aparecen − 1. M5 = suma / 16 tracks | menor |

Nueva regla: se descarta M4 < 10; gana el menor M5; desempata M2. M1 queda como métrica informativa.
Sesgo declarado: la auditoría la hace el mismo agente; los recortes se generan con el mismo protocolo para todos.

## Alternativas evaluadas

| ID | Tracker | Configuración |
|---|---|---|
| T1 | ByteTrack (boxmot) | `track_thresh=0.4`, `min_conf=0.1`, `track_buffer=30` |
| T2 | BoT-SORT (boxmot) | CMC activado, **sin** embeddings |
| T3 | BoT-SORT (boxmot) | CMC + Re-ID OSNet x0.25 (MSMT17) |
| T4 | DeepOCSORT (boxmot) | CMC + Re-ID OSNet x0.25 (MSMT17) |

## Método
`spikes/tracker/run_spike.py`. Detecciones cacheadas fuera del repo (`<data>/cache/dets/`). Semilla fija para
k-means. Hardware: GTX 1660 Super 6 GB, Windows 10.

## Resultados (2026-10-02)

Métricas automáticas, 10 clips (fuente: `<data>/cache/reports/spike001_results.json`):

| Tracker | M1 cambios de color (suma) | M2 fragmentación (mediana) | M3 vida media (mediana) | M4 FPS mín / mediana |
|---|---|---|---|---|
| T1 ByteTrack | 425 | 5,03 | 0,116 | 153 / 159 |
| T2 BoT-SORT + CMC | 426 | 3,71 | 0,160 | 45 / 48 |
| T3 BoT-SORT + CMC + Re-ID | 429 | 3,75 | 0,160 | 16,5 / 17 |
| T4 DeepOCSORT + Re-ID | 381 | 7,11 | 0,083 | **3,0 / 3,3** |

Auditoría visual ciega M5 (8 tracks más largos × 2 clips; letras asignadas al azar por clip, mapeo leído
después de puntuar):

| Tracker | A2 | K2 | **M5 total (16 tracks)** | por track |
|---|---|---|---|---|
| T1 ByteTrack | 6 | 9 | **15** | 0,94 |
| T2 BoT-SORT + CMC | 14 | 12 | 26 | 1,63 |
| T3 BoT-SORT + CMC + Re-ID | 9 | 12 | 21 | 1,31 |
| T4 DeepOCSORT + Re-ID | 6 | 10 | 16 | 1,00 |

Observaciones (hechos):
- M1 casi no discrimina (381–429): confirma que no sirve como métrica primaria.
- Hubo intercambios **entre equipos** (líbero blanco #1 → rojo #9 en K2; blanco → oscuro #11 → #22 en A2) y
  muchos entre compañeros del mismo color.
- Todos los trackers, incluido el mejor, intercambian en promedio ~1 vez por track largo en 20 s.
- Los tracks más largos incluyen oficiales, banco y artefactos de borde (no son jugadores en cancha).

## Conclusión y recomendación

- **T4 descartado** por la regla M4 (3 FPS < 10).
- **Gana T1 ByteTrack** (M5 = 15 vs. 21 del siguiente elegible; diferencia > 1). La hipótesis de ADR-0002
  (BoT-SORT con CMC reduce intercambios) **queda refutada** en estos clips: el paneo es suave y la CMC no compensa
  lo que agrega en asociaciones agresivas. ByteTrack fragmenta más (M2 5,0), lo cual es compatible con el diseño:
  cortar es aceptable, mezclar no.
- Incertidumbre: la auditoría es de 16 tracks y la hace el agente; ±2–3 intercambios de ruido por tracker son
  plausibles. Se revalida con IDF1 sobre ground truth en S6.

**Hallazgo de diseño (más importante que la elección):** ningún tracker evita los intercambios. El
`IdentityManager` no puede tratar un tracklet como "una sola persona" de punta a punta: necesita **partir
tracklets** cuando la apariencia (embedding / color) cambia de forma sostenida, antes de asociarlos a identidades.

## Próximos pasos
1. Actualizar ADR-0002: tracker base ByteTrack (parámetros a ajustar en S4).
2. Spec del `IdentityManager` con requisito explícito de partición de tracklets por discontinuidad de apariencia.
3. Filtro de "jugador en cancha" (máscara por color) antes del tracking, para no gastar asociaciones en banco y
   oficiales.
