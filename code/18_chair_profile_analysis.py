"""What kind of neurosurgeon becomes a chair: productivity, subspecialty, funding.

A  Publications before first appointment, by era, and the chair's percentile within
   the publication distribution of their own graduation cohort (non-leader graduates
   from the stratified PubMed sample, 14_pubmed_academic.py).
B  Dominant subspecialty from MeSH topic counts, by era of appointment.
C  NIH awards held as principal investigator before appointment, by era. RePORTER
   covers FY1985 onward, so this is restricted to appointments from 1995.

Run:  python code/18_chair_profile_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

cp = pd.read_csv(PROC / "chair_profile.csv")
cp = cp[cp.pubmed_name.notna()].copy()
ERAS = ["1960-1979", "1980-1989", "1990-1999", "2000-2009", "2010-2026"]
cp["era"] = pd.cut(cp.appointment, [0, 1979, 1989, 1999, 2009, 2100], labels=ERAS)
cp["decade"] = (cp.graduation // 10 * 10).astype(int)

# ---------------------------------------------------------------- A
pos = pd.read_csv(PROC / "positions.csv")
inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
leaders = set(pos[pos.institution_id.isin(NA)].person_id)

parts = [pd.read_csv(PROC / "pubmed_academic.csv")]
# Reference sample for 1945-1969 graduates (19_pubmed_reference_pre1970.py), used when present.
pre = PROC / "pubmed_academic_pre1970.csv"
if pre.exists():
    parts.append(pd.read_csv(pre))
ref = pd.concat(parts, ignore_index=True)
ref = ref[ref.matched.fillna(False) & ~ref.person_id.isin(leaders)].copy()
ref["decade"] = (ref.graduation // 10 * 10).astype(int)
ref_by_dec = {d: g.pubs_ns.fillna(0).to_numpy() for d, g in ref.groupby("decade")}

rows = []
for era, d in cp.groupby("era", observed=True):
    pct = []
    for r in d.itertuples():
        pool = ref_by_dec.get(r.decade)
        if pool is not None and len(pool) >= 25 and not np.isnan(r.pubs_5_15):
            pct.append((pool < r.pubs_5_15).mean() + 0.5 * (pool == r.pubs_5_15).mean())
    rows.append({
        "era": era, "chairs": len(d),
        "pubs_before_median": float(d.pubs_before.median()),
        "pubs_before_q1": float(d.pubs_before.quantile(.25)),
        "pubs_before_q3": float(d.pubs_before.quantile(.75)),
        "pct_with_20plus_before": round(100 * float((d.pubs_before >= 20).mean()), 1),
        "pct_with_none_before": round(100 * float((d.pubs_before == 0).mean()), 1),
        "n_with_cohort": len(pct),
        "percentile_in_cohort": round(float(np.mean(pct)), 3) if pct else np.nan,
    })
A = pd.DataFrame(rows)
A.to_csv(TABLES / "v08_chair_productivity_by_era.csv", index=False)

# Same percentile by decade of appointment, to sit beside the workforce percentiles
# in Fig. 2C. 22_productivity_primary_5_10.py rewrites this table on the years 5-10 window.
rows = []
cp["appt_decade"] = (cp.appointment // 10 * 10).astype(int)
for dec, d in cp.groupby("appt_decade"):
    pct = [(ref_by_dec[r.decade] < r.pubs_5_15).mean()
           + 0.5 * (ref_by_dec[r.decade] == r.pubs_5_15).mean()
           for r in d.itertuples()
           if r.decade in ref_by_dec and len(ref_by_dec[r.decade]) >= 25
           and not np.isnan(r.pubs_5_15)]
    if len(pct) >= 10:
        rows.append({"decade": int(dec), "n": len(pct),
                     "pct_publications": round(float(np.mean(pct)), 3)})
pd.DataFrame(rows).to_csv(TABLES / "v08_chair_pub_percentile_by_decade.csv", index=False)

# ---------------------------------------------------------------- B
ss_cols = [c for c in cp.columns if c.startswith("ss_")]
ss = cp[ss_cols].fillna(0)
tot = ss.sum(axis=1)
top = ss.idxmax(axis=1).str.replace("ss_", "", regex=False)
share = ss.max(axis=1) / tot.replace(0, np.nan)
cp["subspecialty"] = np.where((tot >= 5) & (ss.max(axis=1) >= 3), top, "unclassified")
cp["ss_share"] = share
B = (pd.crosstab(cp.era, cp.subspecialty, normalize="index") * 100).round(1)
B["n"] = cp.groupby("era", observed=True).size()
B.to_csv(TABLES / "v08_chair_subspecialty_by_era.csv")

# ---------------------------------------------------------------- C
fund = cp[(cp.appointment >= 1995) & cp.nih_before_any.notna()].copy()
rows = []
for era, d in fund.groupby("era", observed=True):
    if len(d) < 10:
        continue
    funded = d[d.nih_before_any.astype(bool)]
    rows.append({
        "era": era, "chairs": len(d),
        "pct_with_prior_NIH": round(100 * float(d.nih_before_any.astype(bool).mean()), 1),
        "pct_exact_first_name": round(100 * float((d.nih_exact_first_n.fillna(0) > 0).mean()), 1),
        "median_awards_if_funded": float(funded.nih_before_n.median()) if len(funded) else np.nan,
        "median_years_first_award_to_chair": float(
            (funded.appointment - funded.nih_first_fy).median()) if len(funded) else np.nan,
    })
C = pd.DataFrame(rows)
C.to_csv(TABLES / "v08_chair_nih_by_era.csv", index=False)

# subspecialty x funding, recent era only
recent = fund[fund.era == "2010-2026"]
D = (recent.groupby("subspecialty")
     .agg(n=("person_id", "size"),
          pct_nih=("nih_before_any", lambda s: round(100 * s.astype(bool).mean(), 1)),
          median_pubs=("pubs_before", "median"))
     .sort_values("n", ascending=False))

pd.set_option("display.width", 220)
print("A. PUBLICATIONS BEFORE FIRST CHAIR APPOINTMENT\n", A.to_string(index=False))
print("\n   cohort reference pools (non-leader graduates, pubs in yrs 5-15):",
      {d: len(v) for d, v in ref_by_dec.items()})
print("\nB. DOMINANT SUBSPECIALTY BY ERA (% of chairs)\n", B.to_string())
print("\n   median share of a chair's topical output in their top subspecialty:",
      round(float(cp.ss_share.median()), 2))
print("\nC. NIH FUNDING BEFORE APPOINTMENT (appointments from 1995)\n", C.to_string(index=False))
print("\nD. RECENT ERA, BY SUBSPECIALTY\n", D.to_string())
