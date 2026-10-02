# Flujo de trabajo

```
 idea / objetivo aprobado
        │
  ¿incertidumbre técnica? ──sí──► /spike  (docs/spikes, time-box, criterios antes de medir)
        │no                                 │
        ▼                                   ▼
      /spec  (docs/specs, AC verificables, check_spec) ◄──┘
        │
  ¿decisión arquitectónica? ──sí──► /adr (docs/adr)
        │
      /issue (GitHub, label type + sprint, DoD)
        │
      rama <N>-<slug>
        │
      /tdd  RED (falla por la razón esperada) → GREEN → REFACTOR → commit "(Refs #N)"
        │
      /pr   verificación local → reviewer → push → PR "Closes #N" + Spec
        │
      CI: lint · tests (ubuntu) · harness (windows) · seguridad · specs · trazabilidad
        │   + verificar qué ejecutó (check_tests_ran)
        ▼
      merge a main  ← aprobación humana
        │
      /sprint-report al cerrar el sprint
```

## Definition of Done (por Issue)

1. Cada AC tiene al menos un test; se observó el RED y el GREEN.
2. CI verde **y verificado**: `check_tests_ran` reporta los tests esperados, sin skips sin Issue.
3. Cobertura ≥ baseline (trinquete).
4. Reviewer sin hallazgos bloqueantes (o justificados por escrito en el PR).
5. Documentación afectada actualizada (spec, ADR, README, configs).
6. PR con `Closes #N` y la spec.

## Modo de ejecución

Sprints largos y autónomos dentro del alcance aprobado. Las dudas no bloqueantes se agrupan al final del sprint.
Se detiene solo la operación que tiene riesgo irreversible, de seguridad, de costo o de desvío de alcance.

## Ramas y commits

- `main` protegida: solo por PR con CI verde. Sin push forzado.
- Rama por Issue: `<N>-<slug>` (sprints de setup: `sN/<slug>`).
- Commits chicos con `(Refs #N)`; el PR cierra con `Closes #N`.
- Atribución: los commits del agente terminan con `Co-Authored-By: Claude …`.
