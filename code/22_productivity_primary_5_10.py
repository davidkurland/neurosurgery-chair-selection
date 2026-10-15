"""Productivity percentile on the years 5-10 window, with 5-15 as sensitivity.

The 5-15 window can overlap the chair appointment for early-era chairs, so part of it
measures output produced as chair. The 5-10 window shortens that overlap. This script
uses 5-10 as the primary measure, with 5-15 as a sensitivity analysis, and writes the
figure inputs and the supplemental table. The overlap itself is tabulated by
24_v09_audit_gaps.py.

Run:  python code/22_productivity_primary_5_10.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC, TABLES = ROOT / "data" / "processed", ROOT / "outputs" / "tables"

w = pd.read_csv(PROC / "pubmed_window_5_10.csv").dropna(subset=["pubs_5_10"])
cp = pd.read_csv(PROC / "chair_profile.csv")
cp = cp[cp.pubmed_name.notna()].merge(w, on="person_id", how="left")
pos = pd.read_csv(PROC / "positions.csv")
inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
leaders = set(pos[pos.institution_id.isin(NA)].person_id)

ref = pd.concat([pd.read_csv(PROC / "pubmed_academic.csv"),
                 pd.read_csv(PROC / "pubmed_academic_pre1970.csv")], ignore_index=True)
ref = ref[~ref.person_id.isin(leaders)].copy()
ref["decade"] = (ref.graduation // 10 * 10).astype(int)
ref10 = ref.merge(w, on="person_id", how="inner")
pool10 = {d: g.pubs_5_10.to_numpy() for d, g in ref10.groupby("decade")}
pool15 = {d: g.pubs_ns.fillna(0).to_numpy()
          for d, g in ref[ref.matched.fillna(False)].groupby("decade")}

ERAS = ["1960-1979", "1980-1989", "1990-1999", "2000-2009", "2010-2026"]
cp["era"] = pd.cut(cp.appointment, [0, 1979, 1989, 1999, 2009, 2100], labels=ERAS)
cp["decade"] = (cp.graduation // 10 * 10).astype(int)
cp["appt_decade"] = (cp.appointment // 10 * 10).astype(int)


def pct(d, col, pools):
    v = [(pools[r.decade] < getattr(r, col)).mean()
         + 0.5 * (pools[r.decade] == getattr(r, col)).mean()
         for r in d.itertuples()
         if r.decade in pools and len(pools[r.decade]) >= 25
         and not np.isnan(getattr(r, col))]
    return (round(100 * float(np.mean(v)), 1), len(v)) if v else (np.nan, 0)


rows = []
for era, d in cp.groupby("era", observed=True):
    p10, n10 = pct(d, "pubs_5_10", pool10)
    p15, n15 = pct(d, "pubs_5_15", pool15)
    rows.append({"era": era, "chairs": len(d),
                 "pubs_before_median": float(d.pubs_before.median()),
                 "pubs_before_q1": float(d.pubs_before.quantile(.25)),
                 "pubs_before_q3": float(d.pubs_before.quantile(.75)),
                 "pct_with_none_before": round(100 * float((d.pubs_before == 0).mean()), 1),
                 "percentile_yrs_5_10": p10, "n_5_10": n10,
                 "percentile_yrs_5_15": p15, "n_5_15": n15})
S7 = pd.DataFrame(rows)
S7.to_csv(TABLES / "v09_productivity_by_era.csv", index=False)
S7.to_csv(TABLES / "supplemental_table_S7_productivity.csv", index=False)

rows = []
for dec, d in cp.groupby("appt_decade"):
    p, k = pct(d, "pubs_5_10", pool10)
    if k >= 10:
        rows.append({"decade": int(dec), "n": k, "pct_publications": round(p / 100, 3)})
pd.DataFrame(rows).to_csv(TABLES / "v08_chair_pub_percentile_by_decade.csv", index=False)

pd.set_option("display.width", 200)
print("PRODUCTIVITY, PRIMARY WINDOW YEARS 5-10 (5-15 as sensitivity)")
print(S7.to_string(index=False))
