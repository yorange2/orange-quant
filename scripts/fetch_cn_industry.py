"""Fetch a SW level-1 industry snapshot into data/cn_industry.csv (roadmap C6).

Sources (akshare → sws 申万官网, the only reliable endpoint on this network;
东财/新浪/历史分类接口均被 SSL 限制):
  * ``sw_index_first_info`` — the 31 level-1 industries;
  * ``index_component_sw`` per industry — current constituents.

SURVIVORSHIP CAVEAT: this is a CURRENT snapshot approximated as history (a
stock's 2026 industry is used for 2017+ labels). Same caveat as the universe
membership snapshot — a stock that changed industry mid-history is tagged
with today's industry. Delisted names are absent (their labels become NaN).

Output: data/cn_industry.csv — header comment + columns code,industry,
industry_code. Codes normalized to the repo's SH/SZ prefix form.

Run from orange-quant/::
    ../.venv/bin/python scripts/fetch_cn_industry.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd


def normalize(code6: str) -> str:
    code6 = code6.strip().zfill(6)
    return f"SH{code6}" if code6.startswith(("6", "9")) else f"SZ{code6}"


def main() -> None:
    info = ak.sw_index_first_info()
    rows = []
    for _, r in info.iterrows():
        ind_code, ind_name = str(r["行业代码"]).split(".")[0], r["行业名称"]
        cons = ak.index_component_sw(symbol=ind_code)
        for c in cons["证券代码"]:
            rows.append({"code": normalize(str(c)), "industry": ind_name,
                         "industry_code": ind_code})
        print(f"[industry] {ind_name}: {len(cons)} names")
    df = pd.DataFrame(rows).drop_duplicates("code").sort_values("code")
    out = Path("data/cn_industry.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# SW level-1 industry snapshot fetched {date.today()}\n"
              "# CURRENT classification approximated as history "
              "(survivorship caveat, same as universe membership)\n")
    out.write_text(header + df.to_csv(index=False))
    print(f"[industry] {len(df)} names × {df['industry'].nunique()} industries "
          f"→ {out}")


if __name__ == "__main__":
    main()
