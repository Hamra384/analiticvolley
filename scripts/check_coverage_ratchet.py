"""Trinquete de cobertura: la cobertura de líneas no puede bajar respecto del baseline versionado.

Sin baseline, falla (un resultado no medido no es aprobado). El baseline solo se actualiza con
--update, como cambio explícito en un PR (quality/baseline.json está protegido por guard_edit).
Uso: python scripts/check_coverage_ratchet.py coverage.xml [--baseline quality/baseline.json] [--update]
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TOLERANCE = 0.005  # medio punto: evita falsos positivos por redondeo


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--baseline", type=Path, default=Path("quality/baseline.json"))
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args(argv)

    if not args.report.is_file():
        print(f"::error::no existe {args.report}")
        return 1
    rate = float(ET.parse(args.report).getroot().get("line-rate", "nan"))
    if args.update:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.is_file() else {}
        data["coverage_line_rate"] = round(rate, 4)
        args.baseline.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"baseline actualizado: {rate:.4f}")
        return 0
    if not args.baseline.is_file():
        print(f"::error::no hay baseline en {args.baseline}: cobertura desconocida, no aprobada")
        return 1
    base = float(json.loads(args.baseline.read_text(encoding="utf-8"))["coverage_line_rate"])
    print(f"check_coverage_ratchet: actual={rate:.4f} baseline={base:.4f}")
    if rate < base - TOLERANCE:
        print(f"::error::la cobertura bajó de {base:.2%} a {rate:.2%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
