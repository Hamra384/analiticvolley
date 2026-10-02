---
name: issue
description: Convierte una spec aprobada en uno o más GitHub Issues acotados y verificables, con labels y referencias a spec y ADR.
---

1. Partí la spec en Issues independientes de alcance chico (idealmente < 1 sesión de trabajo cada uno).
2. Creá cada Issue con `gh issue create --label "type:feature,sprint:SN"` y este cuerpo:
   - **Objetivo** · **Descripción técnica** · **Spec:** `docs/specs/NNN-*.md` · **ADR:** …
   - **Criterios de aceptación** (los AC de la spec que cubre, como checklist)
   - **Tareas de implementación** · **Tareas de testing** · **Dependencias** (#N) · **Riesgos**
   - **Definition of Done:** tests de los AC en verde en CI, `check_tests_ran` OK, reviewer sin hallazgos
     bloqueantes, docs actualizadas, PR con `Closes #N`.
3. Agregá el número de Issue a la spec (`**Issue:** #N`).
4. Rama por Issue: `git checkout -b <N>-<slug>`.
