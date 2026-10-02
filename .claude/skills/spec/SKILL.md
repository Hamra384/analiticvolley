---
name: spec
description: Escribe una especificación técnica y funcional verificable en docs/specs/ a partir de un objetivo aprobado. Usar antes de cualquier implementación de producto.
---

1. Leé `CLAUDE.md`, los ADR relevantes y specs relacionadas. Delegá al subagente `architect` si la spec es grande.
2. Copiá `docs/specs/_template.md` a `docs/specs/NNN-<slug>.md`.
3. Cada criterio de aceptación: `- [ ] AC-N: dado <contexto>, cuando <acción>, entonces <resultado medible>`.
   Nada de "funciona bien" o "es robusto": números, estados, IDs concretos.
4. En "Estrategia de testing" asigná cada AC a un tipo de test (unit / escenario / integración / eval).
5. Indicá si hace falta ADR (y crealo con `/adr`) o por qué no.
6. Validá: `uv run python scripts/check_spec.py docs/specs/NNN-<slug>.md` debe devolver 0.
7. Si cambia el alcance más tarde, primero se actualiza la spec, después el código.
