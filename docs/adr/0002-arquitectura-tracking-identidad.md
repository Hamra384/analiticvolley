# ADR-0002: Separar tracking de corto plazo e identidad persistente

- **Estado:** aceptado (la elección concreta de tracker queda sujeta al spike de S2)
- **Fecha:** 2026-10-01
- **Issue:** #1

## Contexto y problema
El prototipo usa el `track_id` de ByteTrack como identidad del jugador. Una oclusión o un fallo del detector crea
un track nuevo y, con él, un "jugador nuevo". El MVP exige `PLAYER ID ≠ TRACK ID` y re-identificación tras
oclusiones, superposiciones y sustituciones, sin depender del número de camiseta.

Hechos de la auditoría de videos (2026-10-01):
- Los videos de evaluación son edición solo-rallies (185–198 cortes por partido, tramos de ~15 s de mediana).
- La cámara de cabecera hace paneo (evidencia visual: el poste de red entra y sale del cuadro).
- Cada equipo tiene un líbero con color contrastante.
- Altura mediana de jugador: ~190 px (1080p) y ~130 px (720p); el equipo lejano tiene números ilegibles.

## Decisión
Pipeline en capas con un `IdentityManager` propio, puro Python y testeable sin video:

```
VideoSource → ShotDetector (cortes) → PlayerDetector / BallDetector
  → CourtMask (por color, por video) → Tracker (tracklets cortos, con compensación de movimiento de cámara)
  → Appearance (embedding Re-ID + color) | TeamClassifier (colores de equipo y líbero por config + lado de red)
  | JerseyReader (metadata, nunca clave de identidad)
  → IdentityManager (tracklet → PlayerIdentity; máquina de estados; Re-ID; sustituciones)
  → BallTracker (Kalman + gating; DETECTED/TRACKED/PREDICTED/LOST/REACQUIRED)
  → Output JSONL por frame + video de debug
```

- El tracker produce tracklets de corto plazo; ante duda, corta. Unir tracklets es decisión del `IdentityManager`.
- La identidad combina equipo (condición dura), movimiento, apariencia, dorsal (fuerte si es confiable, nunca veto
  absoluto) y contexto temporal. Asignación por algoritmo húngaro.
- Un tracklet nuevo **nunca** crea una identidad automáticamente: solo si ninguna identidad
  `OCCLUDED`/`LOST` es compatible y hay evidencia de jugador nuevo.
- Se procesa por tramo (entre cortes). La identidad a través de cortes es métrica secundaria, no criterio de éxito.
- Tracker base: BoT-SORT (incluye compensación de movimiento de cámara). Se compara contra ByteTrack y DeepOCSORT
  en el spike de S2; si pierde, este ADR se actualiza.
- Se descarta la homografía fija por video (la cámara se mueve). La cancha se delimita por color.

## Alternativas evaluadas
| Alternativa | A favor | En contra |
|---|---|---|
| Identidad = track_id (prototipo) | Simple | Viola el requisito central; IDs nuevas tras cada oclusión |
| Tracker con Re-ID integrado como única capa (StrongSORT/BoT-SORT-ReID) | Menos código | Sin cupo por equipo, sin dorsal, sin sustituciones; difícil de testear |
| **Tracker corto + IdentityManager propio** | Testeable con escenarios sintéticos; combina señales | Más código propio |

## Consecuencias
- Positivas: los 10 tests críticos del MVP se pueden escribir como escenarios sintéticos deterministas en CI.
- Negativas: hay que mantener la lógica de asociación propia.

## Riesgos
- Uniformes similares entre equipos y líberos: mitigado con config por video.
- Tramos cortos (~15 s) limitan la historia disponible para Re-ID.

## Criterio de revisión
Si en S6 el IDF1 queda por debajo del objetivo y el análisis de errores apunta al diseño de capas (y no a un
componente), se reabre.
