"""Women's chair appointments against three pipeline denominators.

Denominators for "who was eligible to be appointed chair":
  all        every North American graduate
  leader     graduates who ever held a chair or PD role   (circular; a bound)
  academic   graduates with >=3 neurosurgery publications in years 5-15 after
             graduation, from the stratified PubMed sample (14_pubmed_academic.py)

For the academic denominator the sex-specific academic proportions estimated in the
sample are applied to the complete NeurosurGen graduate counts, so the era pipeline
sizes remain those of the full cohort. Sampling error in those proportions is
propagated by a stratified bootstrap.

Run:  python code/15_v05_women_denominators.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
RNG = np.random.default_rng(15)
N_SIM = 20000
N_BOOT = 400
THRESH = 3

inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
per = pd.read_csv(PROC / "persons.csv")
pos = pd.read_csv(PROC / "positions.csv")

# Unit = person x host program, matching the placement network used in Table 2.
chair = (pos[(pos.designation == "chair") & pos.institution_id.isin(NA)]
         .dropna(subset=["start"])
         .sort_values("start")
         .drop_duplicates(["person_id", "institution_id"]))
chair = chair.merge(per[["person_id", "gender"]], on="person_id", how="left")
print(f"chair appointments (person x program): {len(chair)}")

g = per[per.residency_id.isin(NA) & per.graduation.notna()].copy()
g["female"] = g.gender == "female"
leaders = set(pos[pos.institution_id.isin(NA)].person_id)
g["leader"] = g.person_id.isin(leaders)
g["decade"] = (g.graduation // 10 * 10).astype(int)

pm = pd.read_csv(PROC / "pubmed_academic.csv")
pm = pm[pm.matched.fillna(False)].copy()
pm["academic"] = pm.pubs_ns.fillna(0) >= THRESH
pm["female"] = pm.gender == "female"
pm["decade"] = (pm.graduation // 10 * 10).astype(int)


def academic_rates(sample):
    """P(academic | sex, graduation decade), with a pooled-by-sex fallback."""
    r = sample.groupby(["female", "decade"]).academic.mean()
    pooled = sample.groupby("female").academic.mean()
    return r, pooled


def female_share_series(weights):
    """Weighted share of women among graduates, by graduation year, 5-year window."""
    w = np.asarray(weights, dtype=float)
    agg = (pd.DataFrame({"graduation": g.graduation.to_numpy(),
                         "f": w * g.female.to_numpy(),
                         "n": w})
           .groupby("graduation").sum())
    years = np.arange(int(agg.index.min()), int(agg.index.max()) + 1)
    agg = agg.reindex(years, fill_value=0.0)
    return (agg.f.rolling(5, center=True, min_periods=1).sum()
            / agg.n.rolling(5, center=True, min_periods=1).sum().replace(0, np.nan))


def weights_for(kind, rates=None, pooled=None):
    if kind == "all":
        return np.ones(len(g))
    if kind == "leader":
        return g.leader.astype(float).to_numpy()
    idx = pd.MultiIndex.from_arrays([g.female, g.decade])
    w = rates.reindex(idx).to_numpy()
    fb = pooled.reindex(g.female).to_numpy()
    return np.where(np.isnan(w), fb, w)


ERA_LAG = {"1990-2009": 14, "2010-present": 16}


def evaluate(series, lag, label, kind):
    rows = []
    e = chair.copy()
    e["era"] = pd.cut(e.start, [0, 1969, 1989, 2009, 2100],
                      labels=["pre-1970", "1970-1989", "1990-2009", "2010-present"])
    groups = [(str(x), e[e.era == x]) for x in ["1990-2009", "2010-present"]]
    groups.append(("1990-present", e[e.start >= 1990]))
    for era, d in groups:
        if lag == "era":
            # each appointment uses its own era's lag, so the pooled 1990-present row is
            # the sum of the two era cells rather than a single 15-year lag
            L = d.era.astype(str).map(ERA_LAG).fillna(15).to_numpy(dtype=float)
        else:
            L = lag
        p = (d.start - L).map(series).to_numpy(dtype=float)
        d = d[~np.isnan(p)]
        p = p[~np.isnan(p)]
        obs = int((d.gender == "female").sum())
        sims = RNG.binomial(1, np.tile(p, (N_SIM, 1))).sum(1)
        rows.append({"denominator": kind, "lag": label, "era": era,
                     "appointments": len(d), "women_observed": obs,
                     "women_expected": round(float(p.sum()), 1),
                     "ratio": round(obs / p.sum(), 2) if p.sum() else np.nan,
                     "p_lower": round(float((np.sum(sims <= obs) + 1) / (N_SIM + 1)), 4)})
    return rows


rates, pooled = academic_rates(pm)
series = {
    "all": female_share_series(weights_for("all")),
    "leader": female_share_series(weights_for("leader")),
    "academic": female_share_series(weights_for("academic", rates, pooled)),
}

rows = []
for kind, s in series.items():
    for lag, label in [("era", "era-specific"), (8, "8 y"), (12, "12 y"), (16, "16 y"), (20, "20 y")]:
        rows.append(evaluate(s, lag, label, kind))
out = pd.DataFrame([r for rs in rows for r in rs])
out.to_csv(TABLES / "v05_women_denominators.csv", index=False)

# Bootstrap the academic denominator (stratified by sex x decade).
boot, bootp = [], []
for _ in range(N_BOOT):
    s = pm.groupby(["female", "decade"], observed=True, group_keys=False).sample(
        frac=1.0, replace=True, random_state=int(RNG.integers(1 << 31)))
    r2, p2 = academic_rates(s)
    ser = female_share_series(weights_for("academic", r2, p2))
    res = evaluate(ser, "era", "era-specific", "academic")
    boot.append([x["ratio"] for x in res])
    bootp.append([x["p_lower"] for x in res])
b = np.array(boot, dtype=float)
bp = np.array(bootp, dtype=float)
ci = pd.DataFrame({"era": ["1990-2009", "2010-present", "1990-present"],
                   "ratio_boot_median": np.nanmedian(b, 0).round(2),
                   "denominator_ci_low": np.nanpercentile(b, 2.5, axis=0).round(2),
                   "denominator_ci_high": np.nanpercentile(b, 97.5, axis=0).round(2),
                   "p_combined": np.nanmean(bp, 0).round(3)})
ci.to_csv(TABLES / "v05_women_academic_bootstrap.csv", index=False)

pd.set_option("display.width", 220)
print("ACADEMIC RATE IN SAMPLE (>=%d neurosurgery pubs, yrs 5-15)" % THRESH)
print(pm.groupby("female").academic.agg(["mean", "size"]).round(3).to_string())
print("\nby graduation decade:")
print(pm.groupby(["female", "decade"]).academic.agg(["mean", "size"]).round(3).to_string())
print("\nWOMEN VS PIPELINE, 3 DENOMINATORS\n", out.to_string(index=False))
print("\nACADEMIC DENOMINATOR: ratio interval reflects sampling of the denominator only;\np_combined integrates that over the binomial variation in the observed appointments\n", ci.to_string(index=False))
