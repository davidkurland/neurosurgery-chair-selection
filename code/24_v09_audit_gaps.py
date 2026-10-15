"""Supplementary numbers cited in the manuscript: KM median tenure, self-hire tests,
birth-year coverage, women chairs, publication-window overlap and Fig. 1 coverage.

Each quantity is written to one table (A to v09_tenure_km_median_age45_55.csv, B-G to
v09_audit_gaps.csv), using the same construction as the script that produced the
underlying data.

  A  Kaplan-Meier median tenure for chairs appointed at ages 45-55, by era, read off
     the curves code/16 saved and using the definition code/08 uses for Table 1
     (first time survival reaches 0.5 or below). code/16 reports an interpolated
     median instead.
  B  Self-hire rate before 1970 vs from 1970 on, Fisher exact.
  C  Birth-year coverage among leaders and among other graduates.
  D  Self-hiring per program since 1970: programs with 3 or more successions, and how
     often they appointed their own graduate.
  E  Women's chair appointments before 1990 on the person x program unit of code/15.
  F  Share of the publication window that falls after appointment, for the 5-15 and
     5-10 windows, before 1980 and since 2010, with the median graduation-to-chair
     interval in each group.
  G  Chair appointments not drawn on the Fig. 1 map.

Run:  python code/24_v09_audit_gaps.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
per = pd.read_csv(PROC / "persons.csv")
pos = pd.read_csv(PROC / "positions.csv")
succ = pd.read_csv(TABLES / "lifecycle_successions.csv")

out = []


def rec(key, value, note):
    out.append({"quantity": key, "value": value, "note": note})


# ------------------------------------------------------------------ A
# Read the curves code/16 saves and plots, and take the median the way
# code/08 does for Table 1: the first time survival reaches 0.5 or below.
km = pd.read_csv(TABLES / "v06_tenure_km_age45_55.csv")
rows = []
for era, g in km.groupby("era", sort=False):
    g = g.sort_values("years")
    below = g.loc[g.surv <= 0.5 + 1e-9, "years"]
    rows.append({"era": era, "n": int(g.n.iloc[0]),
                 "km_median_years": float(below.iloc[0]) if len(below) else np.nan})
A = pd.DataFrame(rows)
A.to_csv(TABLES / "v09_tenure_km_median_age45_55.csv", index=False)

# ------------------------------------------------------------------ B
early = succ[succ.start < 1970]
late = succ[succ.start >= 1970]
tab = [[int(early.own_graduate.sum()), len(early) - int(early.own_graduate.sum())],
       [int(late.own_graduate.sum()), len(late) - int(late.own_graduate.sum())]]
odds, p_fisher = fisher_exact(tab)
rec("selfhire_pre1970_vs_later_p", round(float(p_fisher), 4),
    f"Fisher exact on {tab}; {100*tab[0][0]/sum(tab[0]):.1f}% vs {100*tab[1][0]/sum(tab[1]):.1f}%")
rec("selfhire_pre1970_vs_later_OR", round(float(odds), 2), "odds ratio, same table")

# ------------------------------------------------------------------ C
leaders = set(pos[pos.institution_id.isin(NA)].person_id)
grads = per[per.residency_id.isin(NA) & per.graduation.notna()]
cov = grads.assign(L=grads.person_id.isin(leaders)).groupby("L").birth.apply(
    lambda s: round(100 * float(s.notna().mean()), 1))
rec("birth_year_coverage_leaders_pct", float(cov.get(True, np.nan)),
    f"{int(grads.person_id.isin(leaders).sum())} graduates who ever led a program")
rec("birth_year_coverage_other_graduates_pct", float(cov.get(False, np.nan)),
    f"{int((~grads.person_id.isin(leaders)).sum())} other graduates")

# ------------------------------------------------------------------ D
prog = pd.read_csv(TABLES / "v06_selfhire_by_program.csv")   # already 1970 onward
q = prog[prog.successions >= 3].copy()
q["share"] = q.own_graduate / q.successions
rec("programs_3plus_successions_since_1970", int(len(q)), "denominator for the shares below")
rec("pct_programs_never_own_graduate", round(100 * float((q.own_graduate == 0).mean()), 1), "")
rec("pct_programs_third_or_more_own", round(100 * float((q.share >= 1 / 3).mean()), 1), "")
rec("n_programs_third_or_more_own", int((q.share >= 1 / 3).sum()), "")
rec("n_programs_never_own_graduate", int((q.own_graduate == 0).sum()), "")
rec("pct_programs_two_or_more_own", round(100 * float((q.own_graduate >= 2).mean()), 1),
    "appointed their own graduate more than once")
top10 = prog.sort_values("own_graduate", ascending=False).head(10)
rec("top10_programs_share_of_selfhires_pct",
    round(100 * float(top10.own_graduate.sum() / prog.own_graduate.sum()), 1),
    f"{int(top10.own_graduate.sum())} of {int(prog.own_graduate.sum())} self-hires since 1970")

# ------------------------------------------------------------------ E
chair = (pos[(pos.designation == "chair") & pos.institution_id.isin(NA)]
         .dropna(subset=["start"]).sort_values("start")
         .drop_duplicates(["person_id", "institution_id"])
         .merge(per[["person_id", "gender"]], on="person_id", how="left"))
gr = per[per.residency_id.isin(NA) & per.graduation.notna()]
share = (gr.assign(f=(gr.gender == "female").astype(float))
         .groupby(gr.graduation.astype(int)).f.mean())
share = share.rolling(5, center=True, min_periods=1).mean()   # 5-year rolling window
for era, lo, hi, lag in [("pre-1970", 0, 1969, 9), ("1970-1989", 1970, 1989, 11)]:
    d = chair[chair.start.between(lo, hi)]
    p = (d.start - lag).map(share).to_numpy(dtype=float)
    d, p = d[~np.isnan(p)], p[~np.isnan(p)]
    obs = int((d.gender == "female").sum())
    rec(f"women_chairs_{era}", f"{obs} of {len(d)}",
        f"expected {p.sum():.1f} from the all-graduate pipeline {lag} y earlier "
        f"(person x program unit, matching code/15)")

# ------------------------------------------------------------------ F
cp = pd.read_csv(PROC / "chair_profile.csv")
cp = cp[cp.graduation.notna() & cp.appointment.notna()]
for w0, w1 in [(5, 15), (5, 10)]:
    for label, m in [("before 1980", cp.appointment < 1980), ("since 2010", cp.appointment >= 2010)]:
        d = cp[m]
        after = np.clip(d.graduation + w1 - d.appointment, 0, w1 - w0)
        rec(f"pub_window_{w0}_{w1}_pct_after_appointment_{label.replace(' ', '_')}",
            round(100 * float((after / (w1 - w0)).mean()), 1), f"n = {len(d)} chairs")

for label, m in [("before_1980", cp.appointment < 1980), ("since_2010", cp.appointment >= 2010)]:
    rec(f"median_graduation_to_chair_years_{label}",
        float((cp[m].appointment - cp[m].graduation).median()), f"n = {int(m.sum())} chairs")

# ------------------------------------------------------------------ G
e = pd.read_csv(TABLES / "placement_edges_chair.csv")
e = e[e.source.isin(NA) & e.target.isin(NA)]
e_all = e.copy()
e = e.dropna(subset=["source_lat", "source_lon", "target_lat", "target_lon"])
box = lambda lat, lon: lat.between(24, 56) & lon.between(-130, -60)
e = e[box(e.source_lat, e.source_lon) & box(e.target_lat, e.target_lon)]
rec("fig1_appointments_not_drawn", int(len(e_all) - len(e)),
    f"of {len(e_all)} North American chair appointments; no coordinates or off-map")

G = pd.DataFrame(out)
G.to_csv(TABLES / "v09_audit_gaps.csv", index=False)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 90)
print("A. KM MEDIAN TENURE, CHAIRS APPOINTED AGED 45-55\n", A.to_string(index=False))
print("\nB-G. SUPPLEMENTARY NUMBERS\n", G.to_string(index=False))
