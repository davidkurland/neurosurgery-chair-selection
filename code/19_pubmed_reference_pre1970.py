"""Extend the non-chair publication reference to graduates from 1945-1969.

14_pubmed_academic.py sampled graduates from 1970 on, which leaves the earliest chair cohorts
without a cohort to be compared against. Same measure: neurosurgical publications in
years 5-15 after residency graduation.

Run:  python code/19_pubmed_reference_pre1970.py
"""

import importlib.util
from pathlib import Path

import pandas as pd
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
spec = importlib.util.spec_from_file_location("p14", Path(__file__).resolve().parent / "14_pubmed_academic.py")
p14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p14)

per = pd.read_csv(PROC / "persons.csv")
inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
g = per[per.residency_id.isin(NA) & per.graduation.notna()]
g = g[(g.graduation >= 1945) & (g.graduation <= 1969)].copy()
g["era"] = (g.graduation // 10 * 10).astype(int)
print(f"pre-1970 graduates: {len(g)}", flush=True)

out = []
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, rec in enumerate(ex.map(p14.one, g.to_dict("records")), 1):
        out.append(rec)
        if i % 200 == 0:
            print(f"  {i}/{len(g)}", flush=True)
df = pd.DataFrame(out).merge(g[["person_id", "gender", "residency_id", "era"]],
                             on="person_id", how="left")
df.to_csv(PROC / "pubmed_academic_pre1970.csv", index=False)
print(f"\nwrote pubmed_academic_pre1970.csv ({len(df)} rows); "
      f"median pubs yrs 5-15 = {df.pubs_ns.median()}")
