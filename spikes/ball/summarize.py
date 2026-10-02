"""Resume spike002_results.json por detector."""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict

from spikes.common import cache_dir

rows = json.loads((cache_dir("reports") / "spike002_results.json").read_text(encoding="utf-8"))
agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
for r in rows:
    for k in ("P1_coverage", "P2_plausibility", "P3_jumps", "frames_with_candidate", "P5_fps"):
        agg[r["detector"]][k].append(r[k])
print(f"{'det':4s} {'P1 med':>7s} {'P1 min':>7s} {'P2 med':>7s} {'P3 sum':>7s} {'cand':>6s} {'fps':>6s}")
for d, a in agg.items():
    print(
        f"{d:4s} {st.median(a['P1_coverage']):7.3f} {min(a['P1_coverage']):7.3f} "
        f"{st.median(a['P2_plausibility']):7.3f} {sum(a['P3_jumps']):7.0f} "
        f"{st.median(a['frames_with_candidate']):6.3f} {st.median(a['P5_fps']):6.1f}"
    )
for r in rows:
    print(r["clip"], r["detector"], r["P1_coverage"], r["P2_plausibility"], r["P3_jumps"])
