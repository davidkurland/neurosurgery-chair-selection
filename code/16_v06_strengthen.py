"""Career stage, age-adjusted tenure, self-hiring concentration, classifier validation,
and women among program directors.

A  Career stage of new chairs vs the active workforce, by era. Tests whether the rise
   in age and years-since-graduation reflects selection or merely an aging specialty.
B  Does tenure shortening survive adjustment for age at appointment?
C  Is self-hiring broad across programs or concentrated in a few?
D  Validation of the PubMed academic classifier against known academics.
E  Women among program directors vs the pipeline (better powered than chairs).

Run:  python code/16_v06_strengthen.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
RNG = np.random.default_rng(16)
N_SIM = 10000

inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
per = pd.read_csv(PROC / "persons.csv")
pos = pd.read_csv(PROC / "positions.csv")
ERAS = ["<1950", "1950-1969", "1970-1989", "1990-2009", "2010-2026"]
CUTS = [0, 1949, 1969, 1989, 2009, 2100]

grads = per[per.residency_id.isin(NA) & per.graduation.notna()].copy()
chair = (pos[(pos.designation == "chair") & pos.institution_id.isin(NA)]
         .dropna(subset=["start"]).sort_values("start"))
first_chair = chair.drop_duplicates("person_id").merge(
    per[["person_id", "birth", "graduation", "gender"]], on="person_id", how="left")
first_chair = first_chair[first_chair.graduation.notna()]

gy = grads.graduation.to_numpy(float)
gb = grads.birth.to_numpy(float)
gd = grads.death.to_numpy(float)


# ---------------------------------------------------------------- A
def workforce(y, need_birth=False):
    """Graduates plausibly in practice in year y: 5-40 years out, not yet dead."""
    m = (gy <= y - 5) & (gy >= y - 40) & ~(gd < y)
    if need_birth:
        m &= ~np.isnan(gb)
    return m


rows = []
for era, d in first_chair.assign(
        era=pd.cut(first_chair.start, CUTS, labels=ERAS)).groupby("era", observed=True):
    exp_exp, pct_exp, exp_age, pct_age = [], [], [], []
    for r in d.itertuples():
        m = workforce(r.start)
        if m.sum() >= 20:
            pool = r.start - gy[m]
            exp_exp.append(pool.mean())
            pct_exp.append((pool < (r.start - r.graduation)).mean())
        if not np.isnan(r.birth):
            mb = workforce(r.start, need_birth=True)
            if mb.sum() >= 20:
                pool_a = r.start - gb[mb]
                exp_age.append(pool_a.mean())
                pct_age.append((pool_a < (r.start - r.birth)).mean())
    rows.append({
        "era": era, "first_appointments": len(d),
        "obs_yrs_since_grad_mean": round(float((d.start - d.graduation).mean()), 1),
        "workforce_yrs_since_grad_mean": round(float(np.mean(exp_exp)), 1),
        "percentile_in_workforce_experience": round(float(np.mean(pct_exp)), 3),
        "n_with_birth": int(d.birth.notna().sum()),
        "obs_age_mean": round(float((d.start - d.birth).mean()), 1) if d.birth.notna().any() else np.nan,
        "workforce_age_mean": round(float(np.mean(exp_age)), 1) if exp_age else np.nan,
        "percentile_in_workforce_age": round(float(np.mean(pct_age)), 3) if pct_age else np.nan,
    })
A = pd.DataFrame(rows)
A.to_csv(TABLES / "v06_career_stage_vs_workforce.csv", index=False)

birth_cov = (grads.assign(dec=(grads.graduation // 10 * 10).astype(int))
             .groupby("dec").birth.apply(lambda s: round(s.notna().mean(), 2)))
leaders = set(pos[pos.institution_id.isin(NA)].person_id)
cov_leader = grads.assign(L=grads.person_id.isin(leaders)).groupby("L").birth.apply(
    lambda s: round(s.notna().mean(), 3))


# ---------------------------------------------------------------- B
ten = pd.read_csv(TABLES / "lifecycle_tenures.csv").merge(
    per[["person_id", "birth"]], on="person_id", how="left")
ten["age_start"] = ten.start - ten.birth
ten["decade_c"] = (ten.start - 1970) / 10


def cox(df, cols):
    d = df.dropna(subset=cols + ["years", "event"])
    d = d[d.years > 0]
    m = sm.PHReg(d.years, d[cols], status=d.event, ties="breslow")
    r = m.fit(groups=d.institution_id)
    return pd.DataFrame({
        "term": cols, "HR": np.exp(r.params).round(3),
        "CI_low": np.exp(r.conf_int()[:, 0]).round(3),
        "CI_high": np.exp(r.conf_int()[:, 1]).round(3),
        "p": r.pvalues.round(4), "n": len(d), "events": int(d.event.sum())})


B = pd.concat([
    cox(ten, ["decade_c"]).assign(model="all tenures, unadjusted"),
    cox(ten[ten.age_start.notna()], ["decade_c"]).assign(model="with birth year, unadjusted"),
    cox(ten, ["decade_c", "age_start"]).assign(model="adjusted for age at appointment"),
])
B.to_csv(TABLES / "v06_tenure_age_adjusted.csv", index=False)


# ---------------------------------------------------------------- C
s = pd.read_csv(TABLES / "lifecycle_successions.csv")
mod = s[s.start >= 1970]
by_prog = mod.groupby("institution_id").own_graduate.agg(["size", "sum"])
big = by_prog[by_prog["size"] >= 3]
p_common = mod.own_graduate.mean()
obs_var = float(((big["sum"] / big["size"]) ** 2).sum())
sims = []
for _ in range(N_SIM):
    sim = RNG.binomial(big["size"].to_numpy(), p_common)
    sims.append(float(((sim / big["size"].to_numpy()) ** 2).sum()))
sims = np.array(sims)
C = {
    "successions_since_1970": int(len(mod)),
    "programs_with_successions": int(len(by_prog)),
    "programs_with_3plus": int(len(big)),
    "pooled_self_hire_rate": round(float(p_common), 3),
    "programs_3plus_with_no_self_hire_pct": round(100 * float((big["sum"] == 0).mean()), 1),
    "programs_3plus_with_1_self_hire_pct": round(100 * float((big["sum"] == 1).mean()), 1),
    "programs_3plus_with_2plus_self_hires_pct": round(100 * float((big["sum"] >= 2).mean()), 1),
    "share_of_self_hires_from_top10_programs": round(
        float(big["sum"].nlargest(10).sum() / mod.own_graduate.sum()), 3),
    "overdispersion_stat_observed": round(obs_var, 3),
    "overdispersion_stat_null_mean": round(float(sims.mean()), 3),
    "overdispersion_p": round(float((np.sum(sims >= obs_var) + 1) / (N_SIM + 1)), 4),
}


# ---------------------------------------------------------------- D
pm = pd.read_csv(PROC / "pubmed_academic.csv")
pm = pm[pm.matched.fillna(False)].copy()
pm["leader"] = pm.person_id.isin(leaders)
rows = []
for thr in (1, 3, 5):
    a = pm.pubs_ns.fillna(0) >= thr
    rows.append({"threshold": thr,
                 "sensitivity_among_chairs_or_PDs": round(float(a[pm.leader].mean()), 3),
                 "n_leaders": int(pm.leader.sum()),
                 "rate_among_non_leaders": round(float(a[~pm.leader].mean()), 3),
                 "n_non_leaders": int((~pm.leader).sum())})
D = pd.DataFrame(rows)
D.to_csv(TABLES / "v06_classifier_validation.csv", index=False)


# ---------------------------------------------------------------- E
grads2 = grads.copy()
grads2["female"] = grads2.gender == "female"
grads2["leader"] = grads2.person_id.isin(leaders)
grads2["decade"] = (grads2.graduation // 10 * 10).astype(int)
pm["female"] = pm.gender == "female"
pm["decade"] = (pm.graduation // 10 * 10).astype(int)
pm["academic"] = pm.pubs_ns.fillna(0) >= 3
rate = pm.groupby(["female", "decade"]).academic.mean()
pooled = pm.groupby("female").academic.mean()


def share_series(w):
    w = np.asarray(w, float)
    agg = (pd.DataFrame({"g": grads2.graduation.to_numpy(), "f": w * grads2.female.to_numpy(), "n": w})
           .groupby("g").sum())
    yrs = np.arange(int(agg.index.min()), int(agg.index.max()) + 1)
    agg = agg.reindex(yrs, fill_value=0.0)
    return (agg.f.rolling(5, center=True, min_periods=1).sum()
            / agg.n.rolling(5, center=True, min_periods=1).sum().replace(0, np.nan))


idx = pd.MultiIndex.from_arrays([grads2.female, grads2.decade])
w_ac = np.where(np.isnan(rate.reindex(idx).to_numpy()),
                pooled.reindex(grads2.female).to_numpy(), rate.reindex(idx).to_numpy())
series = {"all": share_series(np.ones(len(grads2))),
          "leader": share_series(grads2.leader.astype(float)),
          "academic": share_series(w_ac)}

pdp = (pos[(pos.designation == "pd") & pos.institution_id.isin(NA)]
       .dropna(subset=["start"]).sort_values("start")
       .drop_duplicates(["person_id", "institution_id"])
       .merge(per[["person_id", "gender"]], on="person_id", how="left"))
# Years from graduation to PD appointment; the pooled row uses a single lag.
PD_LAG = {"1990-2009": 14, "2010-present": 10, "1990-present": 12}
rows = []
for kind, ser in series.items():
    for era, d in [("1990-2009", pdp[(pdp.start >= 1990) & (pdp.start <= 2009)]),
                   ("2010-present", pdp[pdp.start >= 2010]),
                   ("1990-present", pdp[pdp.start >= 1990])]:
        p = (d.start - PD_LAG[era]).map(ser).to_numpy(float)
        d2 = d[~np.isnan(p)]
        p = p[~np.isnan(p)]
        obs = int((d2.gender == "female").sum())
        sims = RNG.binomial(1, np.tile(p, (N_SIM, 1))).sum(1)
        rows.append({"denominator": kind, "era": era, "pd_appointments": len(d2),
                     "women_observed": obs, "women_expected": round(float(p.sum()), 1),
                     "ratio": round(obs / p.sum(), 2) if p.sum() else np.nan,
                     "p_lower": round(float((np.sum(sims <= obs) + 1) / (N_SIM + 1)), 4)})
E = pd.DataFrame(rows)
E.to_csv(TABLES / "v06_women_pd_pipeline.csv", index=False)

pd.set_option("display.width", 240)
print("A. CAREER STAGE OF NEW CHAIRS VS THE ACTIVE WORKFORCE\n", A.to_string(index=False))
print("\n   birth-year coverage by graduation decade:\n  ", birth_cov.to_dict())
print("   birth-year coverage, non-leaders vs leaders:", cov_leader.to_dict())
print("\nB. TENURE, ADJUSTED FOR AGE AT APPOINTMENT\n",
      B[["model", "term", "HR", "CI_low", "CI_high", "p", "n", "events"]].to_string(index=False))
print("\nC. SELF-HIRING ACROSS PROGRAMS (successions since 1970)")
for k, v in C.items():
    print(f"   {k}: {v}")
print("\nD. PUBMED CLASSIFIER VALIDATION\n", D.to_string(index=False))
print("\nE. WOMEN AMONG PROGRAM DIRECTORS VS PIPELINE\n", E.to_string(index=False))


# ---------------------------------------------------------------- F
# Tables that feed the figures.

# F1. Workforce experience/age distribution by calendar year, plus the chairs'
#     mean percentile by decade of appointment.
rows = []
for y in range(1920, 2027):
    m = workforce(y)
    mb = workforce(y, need_birth=True)
    if m.sum() < 20:
        continue
    exp = y - gy[m]
    r = {"year": y, "n_workforce": int(m.sum())}
    for q, lab in [(25, "p25"), (50, "p50"), (75, "p75")]:
        r[f"exp_{lab}"] = round(float(np.percentile(exp, q)), 2)
    if mb.sum() >= 20:
        age = y - gb[mb]
        for q, lab in [(25, "p25"), (50, "p50"), (75, "p75")]:
            r[f"age_{lab}"] = round(float(np.percentile(age, q)), 2)
    rows.append(r)
wf = pd.DataFrame(rows)
wf.to_csv(TABLES / "v06_workforce_by_year.csv", index=False)

rows = []
fc = first_chair.copy()
fc["decade"] = (fc.start // 10 * 10).astype(int)
for dec, d in fc[fc.decade >= 1920].groupby("decade"):
    pe, pa = [], []
    for r in d.itertuples():
        m = workforce(r.start)
        if m.sum() >= 20:
            pe.append(((r.start - gy[m]) < (r.start - r.graduation)).mean())
        if not np.isnan(r.birth):
            mb = workforce(r.start, need_birth=True)
            if mb.sum() >= 20:
                pa.append(((r.start - gb[mb]) < (r.start - r.birth)).mean())
    if len(d) >= 10:
        rows.append({"decade": int(dec), "n": len(d),
                     "pct_experience": round(float(np.mean(pe)), 3) if pe else np.nan,
                     "pct_age": round(float(np.mean(pa)), 3) if pa else np.nan})
pctd = pd.DataFrame(rows)
pctd.to_csv(TABLES / "v06_chair_percentile_by_decade.csv", index=False)


# F2. Kaplan-Meier tenure curves for chairs appointed at ages 45-55.
def km(times, events):
    t = np.sort(np.unique(times[events == 1]))
    s, out = 1.0, [(0.0, 1.0)]
    for ti in t:
        at_risk = (times >= ti).sum()
        d = ((times == ti) & (events == 1)).sum()
        if at_risk:
            s *= 1 - d / at_risk
        out.append((float(ti), s))
    return pd.DataFrame(out, columns=["years", "surv"])


band = ten[(ten.age_start >= 45) & (ten.age_start <= 55)].dropna(subset=["years", "event"])
rows = []
for era, d in band.groupby("era", observed=True):
    if len(d) < 20:
        continue
    c = km(d.years.to_numpy(float), d.event.to_numpy(int))
    c["era"] = era
    c["n"] = len(d)
    rows.append(c)
kmb = pd.concat(rows)
kmb.to_csv(TABLES / "v06_tenure_km_age45_55.csv", index=False)
band_med = band.groupby("era", observed=True).apply(
    lambda d: pd.Series({"n": len(d), "median": float(np.interp(
        -0.5, -km(d.years.to_numpy(float), d.event.to_numpy(int)).surv.to_numpy(),
        km(d.years.to_numpy(float), d.event.to_numpy(int)).years.to_numpy()))}),
    include_groups=False)


# F3. Women's PD appointments by the 4 eras used in Fig. 3.
pdp2 = pdp.merge(per[["person_id", "graduation"]], on="person_id", how="left")
pdp2["era"] = pd.cut(pdp2.start, [0, 1969, 1989, 2009, 2100],
                     labels=["pre-1970", "1970-1989", "1990-2009", "2010-present"])
lag_by_era = (pdp2.assign(l=pdp2.start - pdp2.graduation)
              .groupby("era", observed=True).l.median().round().to_dict())
rows = []
for era, d in pdp2.groupby("era", observed=True):
    L = lag_by_era.get(era, 12)
    p = (d.start - L).map(series["all"]).to_numpy(float)
    d2 = d[~np.isnan(p)]
    p = p[~np.isnan(p)]
    rows.append({"era": str(era), "lag_years": L, "pd_appointments": len(d2),
                 "women_observed": int((d2.gender == "female").sum()),
                 "women_expected": round(float(p.sum()), 2)})
PD4 = pd.DataFrame(rows)
PD4.to_csv(TABLES / "v06_women_pd_4era.csv", index=False)


# F4. Self-hiring per program since 1970.
prog = (mod.groupby("institution_id").own_graduate.agg(successions="size", own_graduate="sum")
        .reset_index()
        .merge(inst[["institution_id", "institution"]], on="institution_id", how="left")
        .sort_values(["own_graduate", "successions"], ascending=False))
prog.to_csv(TABLES / "v06_selfhire_by_program.csv", index=False)

print("\nF. FIGURE INPUTS")
print("   workforce by year:", len(wf), "years;  chair percentile by decade:\n",
      pctd.to_string(index=False))
print("   KM age 45-55 medians:\n", band_med.to_string())
print("   women PD by 4 eras:\n", PD4.to_string(index=False))
print("   programs since 1970:", len(prog))
