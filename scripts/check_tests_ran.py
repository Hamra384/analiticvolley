"""Verifica qué ejecutó pytest realmente (un pipeline verde con 0 tests no es verde).

Falla si: no existe el reporte JUnit, se ejecutaron 0 tests (o menos que --min-tests), hubo fallos o
errores, o algún test fue salteado sin referencia a un Issue (#N) en el motivo.
Uso: python scripts/check_tests_ran.py junit.xml [--min-tests N]
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ISSUE_REF = re.compile(r"#\d+")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--min-tests", type=int, default=1)
    args = ap.parse_args(argv)

    if not args.report.is_file():
        print(f"::error::no existe el reporte {args.report}: los tests no corrieron")
        return 1
    root = ET.parse(args.report).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    skipped = [
        (case.get("classname", "") + "::" + case.get("name", ""), sk.get("message", ""))
        for s in suites
        for case in s.iter("testcase")
        for sk in case.findall("skipped")
    ]
    unjustified = [(n, m) for n, m in skipped if not ISSUE_REF.search(m)]

    print(f"check_tests_ran: tests={tests} fallos+errores={failures} salteados={len(skipped)}")
    ok = True
    if tests < args.min_tests:
        print(f"::error::se ejecutaron {tests} tests (mínimo {args.min_tests})")
        ok = False
    if failures:
        print(f"::error::{failures} tests fallaron o dieron error")
        ok = False
    for name, msg in unjustified:
        print(f"::error::test salteado sin Issue de referencia: {name} ({msg!r})")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
