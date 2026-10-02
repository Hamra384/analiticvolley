---
name: spike
description: Investigación técnica con time-box cuando hay incertidumbre real (elegir modelo, tracker, enfoque). Produce docs/spikes/NNN-*.md con criterios de decisión definidos antes de medir.
---

1. Confirmá que hay incertidumbre técnica relevante. Si no, no hagas spike: anotá en la spec por qué se omite.
2. Creá o ubicá el Issue (label `type:spike`). Copiá `docs/spikes/_template.md` a `docs/spikes/NNN-<slug>.md`
   (NNN = siguiente número libre).
3. Completá **antes de medir**: pregunta, time-box, criterio de finalización, criterios de decisión con umbrales.
4. Código experimental en `spikes/<slug>/` o en el scratchpad, nunca en `src/`. Fijá semilla y registrá hardware.
5. Medí. Registrá los resultados como hechos (tablas) y separalos de la interpretación.
6. Cerrá con una recomendación y la lista de spec/ADR/Issues que se derivan.
7. Si el spike excede el time-box, detenete y reportá lo encontrado + qué faltaría.
