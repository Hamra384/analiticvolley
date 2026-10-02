"""Resume spike001_results.json por tracker."""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict

from spikes.common import cache_dir

rows = json.loads((cache_dir("reports") / "spike001_results.json").read_text(encoding="utf-8"))
agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
for r in rows:
    for k in ("M1_color_flips", "M2_fragmentation", "M3_median_life", "M4_fps"):
        agg[r["tracker"]][k].append(r[k])
print(f"{'tracker':22s} {'M1 sum':>6s} {'M2 med':>6s} {'M3 med':>6s} {'FPSmin':>6s} {'FPSmed':>6s} clips")
for t, a in agg.items():
    print(
        f"{t:22s} {sum(a['M1_color_flips']):6.0f} {st.median(a['M2_fragmentation']):6.2f} "
        f"{st.median(a['M3_median_life']):6.3f} {min(a['M4_fps']):6.1f} {st.median(a['M4_fps']):6.1f} "
        f"{len(a['M4_fps'])}"
    )
