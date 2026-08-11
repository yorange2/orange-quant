"""SW level-1 industry map (roadmap C6): data/cn_industry.csv → {code: industry}.

The CSV is a CURRENT snapshot (fetched by scripts/fetch_cn_industry.py from
sws 申万官网, the only reliable endpoint on this network) — approximated as
history, with the same survivorship caveat as the universe membership
snapshot: a stock that changed industry mid-history carries today's tag,
and delisted names are absent. Unmapped codes get NaN labels (dropped at
train time).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = "data/cn_industry.csv"


def load_industry_map(path: str = DEFAULT_PATH) -> dict[str, str]:
    """code (SH/SZ prefix) → SW level-1 industry name."""
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p, comment="#")
    return dict(zip(df["code"], df["industry"]))
