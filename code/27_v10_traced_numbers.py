"""Save numbers cited in the manuscript that other scripts do not tabulate.

-> outputs/tables/v10_traced_numbers.csv  (one row per number)
-> outputs/tables/v10_canadian_programs.csv
-> outputs/tables/v10_successors_outside_pool.csv
-> outputs/tables/v10_academic_rate_by_sex.csv
Run:  python code/27_v10_traced_numbers.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
PROC = ROOT / "data" / "processed"
rows = []

# programs with recorded chair or PD terms, and the Canadian ones
pos = pd.read_csv(PROC / "positions.csv"); inst = pd.read_csv(PROC / "institutions.csv")
terms = pos[pos.designation.isin(["chair", "pd"])]
with_terms = inst[inst.institution_id.isin(terms.institution_id)]
na = with_terms[with_terms.country.isin(["United States", "Canada"])]
rows.append({"number": "programs with recorded chair or PD terms (US and Canada)", "value": len(na)})
rows.append({"number": "institutions listed as training sites", "value": len(inst)})
can = na[na.country == "Canada"][["institution", "location"]]
can.to_csv(TABLES / "v10_canadian_programs.csv", index=False)
rows.append({"number": "Canadian programs with recorded terms", "value": len(can)})

# successors outside the 5-25 year pool window, by era
s = pd.read_csv(TABLES / "lifecycle_successions.csv"); per = pd.read_csv(PROC / "persons.csv")
s["ypg"] = s.start - s.successor.map(per.set_index("person_id").graduation)
s["era4"] = pd.cut(s.start, [0, 1969, 1989, 2009, 2100], labels=["pre-1970", "1970-1989", "1990-2009", "2010-present"])
g = s.dropna(subset=["ypg"]).groupby("era4", observed=False).ypg
out = pd.DataFrame({"successions_with_graduation_year": g.size(), "under_5_years": g.apply(lambda x: int((x < 5).sum())),
                    "over_25_years": g.apply(lambda x: int((x > 25).sum()))})
out["outside_pct"] = (100 * (out.under_5_years + out.over_25_years) / out.successions_with_graduation_year).round(1)
out.to_csv(TABLES / "v10_successors_outside_pool.csv")
post = s[(s.start >= 1970)].dropna(subset=["ypg"])
rows.append({"number": "successors since 1970 outside the 5-25 year pool (%)", "value": round(100 * ((post.ypg < 5) | (post.ypg > 25)).mean(), 1)})

# academic publication rate by sex in the stratified PubMed sample (>=3 neurosurgical publications in years 5-15)
pm = pd.read_csv(PROC / "pubmed_academic.csv")
ac = pm.assign(academic=pm.pubs_ns >= 3).groupby("gender").academic.agg(n="size", academic_n="sum", academic_pct=lambda x: round(100 * x.mean(), 1))
ac.to_csv(TABLES / "v10_academic_rate_by_sex.csv")
for sex, r in ac.iterrows():
    rows.append({"number": f"{sex} graduates in the PubMed sample with >=3 neurosurgical publications (%)", "value": r.academic_pct})
rows.append({"number": "graduates in the PubMed sample", "value": int(len(pm))})

# Supplemental Fig. S2: self-hiring heterogeneity across programs (overdispersion against a common rate)
sh = pd.read_csv(TABLES / "v06_selfhire_by_program.csv"); big = sh[sh.successions >= 3]
p_common = big.own_graduate.sum() / big.successions.sum()
obs_var = np.var(big.own_graduate / big.successions)
rng = np.random.default_rng(7); N = 20000
sims = np.array([np.var(rng.binomial(big.successions.to_numpy(), p_common) / big.successions.to_numpy()) for _ in range(N)])
rows.append({"number": "programs with >=3 successions since 1970", "value": len(big)})
pval = float((np.sum(sims >= obs_var) + 1) / (N + 1))
rows.append({"number": "self-hiring heterogeneity p (variance of program rates vs common-rate simulation, one-sided)", "value": f"<{1 / (N + 1):.5f}" if pval <= 1 / (N + 1) else round(pval, 4)})
rows.append({"number": "pooled self-hire rate among those programs (%)", "value": round(100 * p_common, 1)})


# open-ended chair or PD terms followed by a later same-role holder or the holder's death, and graduates attributed past that point
pos2 = pos.merge(per[["person_id", "death"]], on="person_id", how="left").sort_values(["institution_id", "designation", "start"])
pos2["next_start"] = pos2.groupby(["institution_id", "designation"]).start.shift(-1)
oe = pos2[pos2.end.isna()].copy(); oe["eff_end"] = oe[["next_start", "death"]].min(axis=1); oe = oe[oe.eff_end.notna()]
tl = pd.read_csv(PROC / "training_links.csv"); chairs = set(pos[pos.designation == "chair"].person_id); past = 0; past_chairs = 0
for role in ["chair", "pd"]:
    m = tl.merge(oe[oe.designation == role][["person_id", "institution_id", "eff_end"]], left_on=[f"derived_{role}_id", "residency_id"], right_on=["person_id", "institution_id"], suffixes=("", "_m"))
    bad = m[m.graduation > m.eff_end]; past += len(bad); past_chairs += int(bad.person_id.isin(chairs).sum())
rows.append({"number": "open-ended terms followed by a later same-role holder or the holder's death", "value": len(oe)})
rows.append({"number": "graduates attributed to such a term after that point", "value": past})
rows.append({"number": "of whom later chairs", "value": past_chairs})

# extraction and cohort counts cited in the Abstract and Results (computed here so a shipped table holds each)
import json
hier = json.loads((TABLES / "hierarchy_summary_chair.json").read_text())
fair = json.loads((TABLES / "followup_hierarchy_fair_comparison.json").read_text())
ten = pd.read_csv(TABLES / "lifecycle_tenures.csv"); first = pd.read_csv(TABLES / "lifecycle_first_chair_points.csv")
cm = pd.read_csv(TABLES / "lifecycle_chairmakers.csv"); prod = pd.read_csv(TABLES / "v09_productivity_by_era.csv")
pre = pd.read_csv(PROC / "pubmed_academic_pre1970.csv")
rows += [
    {"number": "neurosurgeons in the extraction", "value": len(per)},
    {"number": "leadership terms (chair and PD)", "value": len(terms)},
    {"number": "chair terms", "value": int((terms.designation == "chair").sum())},
    {"number": "PD terms", "value": int((terms.designation == "pd").sum())},
    {"number": "individuals holding a leadership term", "value": int(terms.person_id.nunique())},
    {"number": "individuals holding a chair at a US or Canadian program", "value": int(ten.person_id.nunique())},
    {"number": "first chair appointments", "value": len(first)},
    {"number": "appointments with a North American training program (Fig. 1)", "value": hier["overall"]["n_positions"]},
    {"number": "external chair placements between US and Canadian programs", "value": hier["n_external_placements"]},
    {"number": "placements ranked by both the inferred order and chair counts", "value": fair["placements_untied_in_both"]},
    {"number": "chairs first appointed from 1960 (publication analysis)", "value": int(prod.chairs.sum())},
    {"number": "chairs descending from Harvey Cushing through recorded lineages", "value": int(cm.loc[cm.name == "Harvey W. Cushing", "all_descendant_chairs"].iloc[0])},
    {"number": "North American graduates 1945-1969 in the PubMed productivity extension", "value": len(pre)},
]

out_rows = pd.DataFrame(rows); out_rows["value"] = [int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v for v in out_rows.value]
out_rows.to_csv(TABLES / "v10_traced_numbers.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False)); print(can.to_string(index=False)); print(out.to_string())
