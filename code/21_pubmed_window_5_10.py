"""Publications in years 5-10 after graduation, for chairs and both reference cohorts.

The years 5-15 window used in 14_pubmed_academic.py, 17_chair_profile_fetch.py and
19_pubmed_reference_pre1970.py can overlap the chair appointment, so part of it measures
output produced as chair. This script counts neurosurgical publications in years 5-10
after graduation, a window that ends before appointment for most chairs, for the chairs
(chair_profile.csv) and for both reference cohorts (pubmed_academic.csv and
pubmed_academic_pre1970.csv).

Run:  python code/21_pubmed_window_5_10.py
"""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
spec = importlib.util.spec_from_file_location("p14", Path(__file__).resolve().parent / "14_pubmed_academic.py")
p14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p14)

per = pd.read_csv(PROC / "persons.csv")
inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])

chairs = pd.read_csv(PROC / "chair_profile.csv")[["person_id", "name", "graduation"]]
ref = pd.concat([pd.read_csv(PROC / "pubmed_academic.csv"),
                 pd.read_csv(PROC / "pubmed_academic_pre1970.csv")],
                ignore_index=True)[["person_id", "name", "graduation"]]
people = (pd.concat([chairs, ref], ignore_index=True)
          .dropna(subset=["name", "graduation"])
          .drop_duplicates("person_id"))
print(f"people needing a 5-10 window: {len(people)}", flush=True)


def one(r):
    au = p14.pubmed_name(r["name"])
    if not au:
        return {"person_id": r["person_id"], "pubs_5_10": None}
    g = int(r["graduation"])
    return {"person_id": r["person_id"],
            "pubs_5_10": p14.count(f"{au}[au] AND {p14.NS_FILTER}", g + 5, g + 10, "w510")}


out = []
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, rec in enumerate(ex.map(one, people.to_dict("records")), 1):
        out.append(rec)
        if i % 300 == 0:
            print(f"  {i}/{len(people)}", flush=True)
            pd.DataFrame(out).to_csv(PROC / "pubmed_window_5_10.csv", index=False)
df = pd.DataFrame(out)
df.to_csv(PROC / "pubmed_window_5_10.csv", index=False)
print(f"\nwrote pubmed_window_5_10.csv ({len(df)} rows); "
      f"median = {df.pubs_5_10.median()}")
