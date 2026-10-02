---
spike: "004"
issue: "#17"
fecha: 2026-10-02
estado: cerrado
---

# SPIKE-004: ¿Qué lector de número de camiseta usar?

## Pregunta
Para vincular número ↔ identidad (#13 líberos, #14 identidades mezcladas) hace falta leer el dorsal en video real
(720p/1080p, transmisión). ¿Qué lector existente alcanza la precisión necesaria?

## Criterio de decisión (fijado antes de medir)
Gana el lector con **mayor cobertura** entre los que logran **precisión ≥ 0,95** en algún umbral de confianza.
Un número equivocado es peor que ninguno (contamina la identidad), por eso la precisión es la condición.

## Método
- Set: 180 recortes de torso de personas grandes (altura ≥ 18 % del frame) de A2, A3, K2 y K5 (tracks ByteTrack,
  un recorte cada 15 frames). **Etiquetado a mano por el agente**: 83 con número legible, 69 sin número visible,
  28 dudosos (excluidos de la evaluación). Fuera del repo: `<datos>/cache/jersey/`.
- Precisión = lecturas correctas / lecturas emitidas (inventar un número donde no hay cuenta como error).
  Cobertura = lecturas correctas / recortes con número.
- `spikes/jersey/eval_readers.py`.

## Resultados

| Lector | Precisión ≥ 0,95 | Mejor punto | A umbral 0,9 |
|---|---|---|---|
| **easyocr 1.7.2, imagen directa (gris, alto 160 px, solo dígitos)** | **sí** | umbral 0,95: 26 lecturas, **P 0,96, cobertura 0,30** | P 0,87, cobertura 0,33 |
| easyocr con CLAHE y dos polaridades | no | — | P 0,72, cobertura 0,16 |
| Clasificador MobileNet del prototipo (`models/dorsal_classifier.onnx`) | no | — | P 0,00 |

## Decisión
**easyocr directo, emitiendo solo lecturas con confianza ≥ 0,95.** La cobertura por recorte (30 %) es suficiente:
un jugador cercano se observa decenas de veces por rally y la identidad exige ≥ 3 lecturas coincidentes (RF-7 de
SPEC-001), lo que eleva la precisión a nivel identidad.

El clasificador del prototipo queda descartado (se entrenó con etiquetas automáticas ruidosas y un split con fuga
entre train/val; ver `legacy/README.md`).

## Limitaciones
- Muestra chica: 26 lecturas al umbral elegido (1 error). La precisión real puede estar entre ~0,8 y ~1,0.
- Etiquetado por el agente, no por un humano; los casos dudosos se excluyeron.
- Solo jugadores grandes (cercanos). Los lejanos casi nunca se leen: es esperable y está bien (preferimos `null`).
- Alternativas no probadas (time-box): PARSeq (reconocimiento de texto de escena), fine-tuning con dorsales
  anotados. Quedan como mejora si la cobertura no alcanza en la evaluación de S6.
