"""Two robustness checks on program size and chair production.

1. Does self-hiring concentrate in the programs that produce the most chairs?
   Among the programs with >=3 successions since 1970 (the Supplemental Fig. S2 set), the
   self-hire rate against chairs trained (all time), and the 10 most prolific programs versus the rest.
   -> outputs/tables/v10_selfhire_vs_prolific.csv

2. Chairs per 100 graduates by era for the era's 10 most prolific programs versus all other programs,
   using the same graduation window as the size-only concentration model (10th-90th percentile of the
   era's chairs' graduation years; 06_robustness.py r3).
   -> outputs/tables/v10_prolific_per_graduate_by_era.csv

Run:  python code/26_v10_prolific_programs_checks.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
PROC = ROOT / "data" / "processed"

# ---- 1. self-hiring versus chair production
sh = pd.read_csv(TABLES / "v06_selfhire_by_program.csv")
rank = pd.read_csv(TABLES / "program_ranking_chair.csv")[["institution_id", "program", "leaders_produced", "leaders_produced_elsewhere", "graduates_listed"]]
m = sh.merge(rank, on="institution_id", how="left")
m = m[m.successions >= 3].copy()
m["rate"] = m.own_graduate / m.successions
rho = spearmanr(m.rate, m.leaders_produced_elsewhere)
rho_size = spearmanr(m.rate, m.graduates_listed)
top = m.nlargest(10, "leaders_produced_elsewhere")
rest = m.drop(top.index)
a, b = int(top.own_graduate.sum()), int(top.successions.sum())
c, d = int(rest.own_graduate.sum()), int(rest.successions.sum())
p = fisher_exact([[a, b - a], [c, d - c]])[1]
out1 = pd.DataFrame([
    {"group": "10 programs that placed the most chairs at other programs (all time)", "programs": len(top), "self_hires": a, "successions": b, "pct": round(100 * a / b, 1)},
    {"group": "other programs with >=3 successions since 1970", "programs": len(rest), "self_hires": c, "successions": d, "pct": round(100 * c / d, 1)},
])
out1.to_csv(TABLES / "v10_selfhire_vs_prolific.csv", index=False)
tests = pd.DataFrame([
    {"statistic": "fisher_exact_p_two_sided", "value": p},
    {"statistic": "spearman_selfhire_rate_vs_chairs_placed_elsewhere", "value": round(rho.correlation, 2)},
    {"statistic": "spearman_selfhire_rate_vs_graduates_listed", "value": round(rho_size.correlation, 2)},
    {"statistic": "programs_compared", "value": len(m)},
])
tests.to_csv(TABLES / "v10_selfhire_vs_prolific_tests.csv", index=False)
print(out1.to_string(index=False)); print(tests.to_string(index=False))

# ---- 2. chairs per 100 graduates by era
e = pd.read_csv(TABLES / "placement_edges_chair.csv")
per = pd.read_csv(PROC / "persons.csv")
inst = pd.read_csv(PROC / "institutions.csv")
na = set(inst[inst.country.isin(["United States", "Canada"])].institution_id)
e = e[e.source.isin(na) & e.target.isin(na)].copy()
grads = per.dropna(subset=["graduation"])
grads = grads[grads.residency_id.isin(na)]
e["era4"] = e.era.replace({"<1950": "pre-1970", "1950-1969": "pre-1970"})
rows = []
for era in ["pre-1970", "1970-1989", "1990-2009", "2010-present"]:
    sub = e[e.era4 == era]
    lo, hi = sub.graduation.quantile([0.1, 0.9])
    g = grads[grads.graduation.between(lo, hi)].groupby("residency_id").size()
    prod = sub.groupby("source").size()
    top10 = prod.nlargest(10).index
    tg, tc = g.reindex(top10).fillna(0).sum(), prod.loc[top10].sum()
    others = [x for x in g.index if x not in top10]
    og, oc = g.reindex(others).fillna(0).sum(), prod.reindex(others).fillna(0).sum()
    rows.append({"era": era, "graduation_window": f"{lo:.0f}-{hi:.0f}",
                 "top10_chairs": int(tc), "top10_graduates": int(tg), "top10_per_100_graduates": round(100 * tc / tg, 1),
                 "other_chairs": int(oc), "other_graduates": int(og), "other_per_100_graduates": round(100 * oc / og, 1),
                 "ratio_top10_to_other": round((tc / tg) / (oc / og), 1), "top10_share_of_chairs_pct": round(100 * tc / (tc + oc), 1)})
out2 = pd.DataFrame(rows)
out2.to_csv(TABLES / "v10_prolific_per_graduate_by_era.csv", index=False)
print(out2.to_string(index=False))
