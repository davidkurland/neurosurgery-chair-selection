"""Is academic leadership inherited through training lineage?

Cohort: graduates of US residency programs with a known graduation year.
Outcome: time from residency graduation to first chair appointment (anywhere).
Censoring: death, the snapshot year, or 40 years after graduation.

Every exposure is time-respecting (uses only events before the trainee's
graduation year) so that the outcome cannot leak into its own predictors:
  chair_maker_prior  earlier trainees of the trainee's chair who had already become chairs
  chair_trainees_prior  earlier trainees of that chair (exposure opportunity / size)
  pd_maker_prior     same, for the program director
  prog_prior_rate    program's chairs produced per graduate before the graduation year
  cushing_lineage    trainee's chair or PD descends from Harvey Cushing

Models (Cox proportional hazards, statsmodels PHReg, Breslow ties)
  M1 between-program: strata = graduation era; SEs clustered by program
  M2 within-program:  strata = residency program; SEs clustered by chair
  M3 M2 + PD maker
  Sensitivity: NeurosurGen's own chair/PD assignment; outcome = chair OR PD;
               logistic "chair within 25 y" for graduates <= 2000 (GEE by program)

Run:  python code/04_leadership_inheritance.py
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
SNAPSHOT_YEAR = 2026
MAX_FU = 40
CUSHING = "93914454-61aa-435d-9afa-4e59deaa60e9"


def prior_counts(df, mentor_col, outcome_year):
    """For each trainee: # earlier trainees of the same mentor, and # of those who
    reached the outcome strictly before this trainee's graduation year."""
    d = df.dropna(subset=[mentor_col])[["person_id", mentor_col, "graduation"]]
    d = d.assign(outcome_year=d.person_id.map(outcome_year))
    rows = []
    for m, g in d.groupby(mentor_col):
        grads = g.graduation.to_numpy()
        oy = g.outcome_year.to_numpy()
        for pid, gy in zip(g.person_id, grads):
            earlier = grads < gy
            rows.append((pid, earlier.sum(), np.sum(earlier & (oy < gy))))
    return pd.DataFrame(rows, columns=["person_id", "trainees_prior", "makers_prior"]).set_index("person_id")


def leader_prior_counts(df, outcome_year):
    """Chair and PD are often the same person, so also count across the union:
    distinct earlier trainees of either of the trainee's leaders (chair or PD)
    and how many of them had become chairs before this trainee graduated."""
    long = pd.concat([df[["person_id", "chair_id", "graduation"]].rename(columns={"chair_id": "mentor"}),
                      df[["person_id", "pd_id", "graduation"]].rename(columns={"pd_id": "mentor"})]).dropna()
    by_mentor = {m: g[["person_id", "graduation"]].drop_duplicates() for m, g in long.groupby("mentor")}
    rows = []
    for r in df[["person_id", "chair_id", "pd_id", "graduation"]].itertuples(index=False):
        ms = {m for m in (r.chair_id, r.pd_id) if isinstance(m, str)}
        if not ms:
            continue
        pool = pd.concat([by_mentor[m] for m in ms]).drop_duplicates("person_id")
        earlier = pool[pool.graduation < r.graduation]
        oy = earlier.person_id.map(outcome_year)
        rows.append((r.person_id, len(earlier), int((oy < r.graduation).sum())))
    return pd.DataFrame(rows, columns=["person_id", "trainees_prior", "makers_prior"]).set_index("person_id")


def build_cohort(link_source="derived", outcome="chair"):
    persons = pd.read_csv(PROC / "persons.csv")
    positions = pd.read_csv(PROC / "positions.csv")
    links = pd.read_csv(PROC / "training_links.csv")
    inst = pd.read_csv(PROC / "institutions.csv")
    us = set(inst.loc[inst.is_us, "institution_id"])

    for role in ("chair", "pd"):
        primary, backup = (f"derived_{role}_id", f"site_{role}_id") if link_source == "derived" \
            else (f"site_{role}_id", f"derived_{role}_id")
        links[f"{role}_id"] = links[primary].fillna(links[backup])

    pos = positions if outcome == "leader" else positions[positions.designation == "chair"]
    first_lead = pos.groupby("person_id").start.min()
    first_chair = positions[positions.designation == "chair"].groupby("person_id").start.min()

    df = persons.merge(links[["person_id", "chair_id", "pd_id"]], on="person_id")
    df = df[df.graduation.notna() & df.residency_id.isin(us)].copy()

    # lineage graph over everyone (trainee -> chair / PD)
    G = nx.DiGraph()
    for r in links.itertuples():
        for m in (r.chair_id, r.pd_id):
            if isinstance(m, str) and m != r.person_id:
                G.add_edge(m, r.person_id)
    cushing_desc = nx.descendants(G, CUSHING) | {CUSHING} if CUSHING in G else set()
    df["cushing_lineage"] = (df.chair_id.isin(cushing_desc) | df.pd_id.isin(cushing_desc)).astype(int)

    c = prior_counts(df, "chair_id", first_chair)
    p = prior_counts(df, "pd_id", first_chair)
    df = df.join(c.add_prefix("chair_"), on="person_id").join(p.add_prefix("pd_"), on="person_id")
    df = df.join(leader_prior_counts(df, first_chair).add_prefix("leader_"), on="person_id")

    # program production before graduation year
    df["first_chair_year"] = df.person_id.map(first_chair)
    prog_rows = []
    for prog, g in df.groupby("residency_id"):
        gy, fy = g.graduation.to_numpy(), g.first_chair_year.to_numpy()
        for pid, y in zip(g.person_id, gy):
            earlier = gy < y
            prog_rows.append((pid, earlier.sum(), np.sum(earlier & (fy < y))))
    prog = pd.DataFrame(prog_rows, columns=["person_id", "prog_grads_prior", "prog_chairs_prior"]).set_index("person_id")
    df = df.join(prog, on="person_id")
    df["prog_prior_rate"] = (df.prog_chairs_prior / df.prog_grads_prior.replace(0, np.nan)).fillna(0)

    # survival time
    df["event_year"] = df.person_id.map(first_lead)
    end = np.minimum(df.death.fillna(SNAPSHOT_YEAR), SNAPSHOT_YEAR)
    end = np.minimum(end, df.graduation + MAX_FU)
    df["event"] = (df.event_year.notna() & (df.event_year <= end)).astype(int)
    df["time"] = np.where(df.event == 1, df.event_year - df.graduation, end - df.graduation)
    df["time"] = df.time.clip(lower=0.5)  # appointment in graduation year (or data error) -> half year
    df = df[df.graduation <= SNAPSHOT_YEAR - 1]

    df["female"] = (df.gender == "female").astype(int)
    df["grad_era"] = pd.cut(df.graduation, [0, 1949, 1969, 1989, 2009, 9999],
                            labels=["<1950", "1950-69", "1970-89", "1990-2009", "2010+"]).astype(str)
    df["grad_year_c"] = (df.graduation - 1990) / 10
    for col in ("chair_makers_prior", "chair_trainees_prior", "pd_makers_prior", "pd_trainees_prior",
                "leader_makers_prior", "leader_trainees_prior"):
        df[col] = df[col].fillna(0)
        df[f"log_{col}"] = np.log1p(df[col])
    df["leader_maker_cat"] = pd.cut(df.leader_makers_prior, [-1, 0, 2, 1e9], labels=["0", "1-2", "3+"]).astype(str)
    df["prog_rate_10"] = df.prog_prior_rate * 10  # per 10 percentage-point increase
    df["has_chair_link"] = df.chair_id.notna().astype(int)
    return df


def cox(df, covars, strata, cluster, label):
    d = df.dropna(subset=covars + ["time", "event", strata, cluster])
    X = d[covars].to_numpy(float)
    mod = PHReg(d.time.to_numpy(float), X, status=d.event.to_numpy(int),
                strata=pd.factorize(d[strata])[0], ties="breslow")
    res = mod.fit()
    # Cluster-robust sandwich V = I^-1 (sum_g U_g U_g') I^-1. statsmodels' own
    # groups= option returns NaN when strata contain no events; those subjects'
    # score residuals are NaN but contribute zero, so zero them explicitly.
    # (Matches statsmodels' robust SEs exactly when no stratum is event-free.)
    U = np.nan_to_num(res.score_residuals)
    g = pd.factorize(d[cluster])[0]
    Ug = np.zeros((g.max() + 1, U.shape[1]))
    np.add.at(Ug, g, U)
    V = res.cov_params() @ (Ug.T @ Ug) @ res.cov_params()
    se = np.sqrt(np.diag(V))
    res_p = 2 * (1 - __import__("scipy").stats.norm.cdf(np.abs(res.params / se)))
    out = pd.DataFrame({
        "model": label, "term": covars,
        "HR": np.exp(res.params),
        "CI_low": np.exp(res.params - 1.96 * se),
        "CI_high": np.exp(res.params + 1.96 * se),
        "p": res_p,
        "n": len(d), "events": int(d.event.sum()),
    })
    return out


def main():
    df = build_cohort()
    df.to_csv(PROC / "inheritance_cohort.csv", index=False)
    lk = df[df.has_chair_link == 1]

    # primary exposure: prior trainees of the trainee's chair or PD who became chairs
    base = ["log_leader_makers_prior", "log_leader_trainees_prior", "female"]
    results = [
        cox(lk, base + ["prog_rate_10", "cushing_lineage"], "grad_era", "residency_id", "M1 between-program (strata: era)"),
        cox(lk, base + ["grad_year_c"], "residency_id", "chair_id", "M2 within-program (strata: program)"),
        cox(lk, ["log_chair_makers_prior", "log_chair_trainees_prior", "log_pd_makers_prior",
                 "log_pd_trainees_prior", "female", "grad_year_c"], "residency_id", "chair_id",
            "M3 within-program, chair and PD separately"),
    ]
    # categorical version for interpretability
    lk_cat = lk.assign(maker_1_2=(lk.leader_maker_cat == "1-2").astype(int),
                       maker_3p=(lk.leader_maker_cat == "3+").astype(int))
    results.append(cox(lk_cat, ["maker_1_2", "maker_3p", "log_leader_trainees_prior", "female", "grad_year_c"],
                       "residency_id", "chair_id", "M2c within-program, categorical"))

    # sensitivity: site-assigned links
    s = build_cohort(link_source="site")
    s = s[s.has_chair_link == 1]
    results.append(cox(s, base + ["grad_year_c"], "residency_id", "chair_id", "S1 site-assigned chair/PD"))
    # sensitivity: outcome chair or PD
    l = build_cohort(outcome="leader")
    l = l[l.has_chair_link == 1]
    results.append(cox(l, base + ["grad_year_c"], "residency_id", "chair_id", "S2 outcome chair or PD"))
    # sensitivity: excluding the pre-1950 founders era
    results.append(cox(lk[lk.graduation >= 1950], base + ["grad_year_c"], "residency_id", "chair_id",
                       "S3 graduates 1950+"))

    # sensitivity: within-program effect by graduation era (continuous and 3+ threshold)
    for lab, sub in [("<1970", lk[lk.graduation < 1970]), ("1970-1989", lk[lk.graduation.between(1970, 1989)]),
                     ("1990+", lk[lk.graduation >= 1990])]:
        sub = sub.assign(maker_1_2=(sub.leader_maker_cat == "1-2").astype(int),
                         maker_3p=(sub.leader_maker_cat == "3+").astype(int))
        # no women became chairs among early graduates -> female not estimable there
        sex = ["female"] if sub.loc[sub.event == 1, "female"].sum() > 0 else []
        results.append(cox(sub, ["log_leader_makers_prior", "log_leader_trainees_prior"] + sex + ["grad_year_c"],
                           "residency_id", "chair_id", f"E within-program, grads {lab}"))
        results.append(cox(sub, ["maker_1_2", "maker_3p", "log_leader_trainees_prior"] + sex + ["grad_year_c"],
                           "residency_id", "chair_id", f"Ec within-program categorical, grads {lab}"))

    cox_tab = pd.concat(results)
    cox_tab.to_csv(TABLES / "inheritance_cox_models.csv", index=False)

    # logistic: ever chair within 25 years, graduates <= 2000
    lg = lk[lk.graduation <= 2000].copy()
    lg["chair25"] = ((lg.event == 1) & (lg.time <= 25)).astype(int)
    gee = smf.gee("chair25 ~ log_leader_makers_prior + log_leader_trainees_prior + female + prog_rate_10 + C(grad_era)",
                  groups="residency_id", data=lg, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = gee.conf_int()
    logit_tab = pd.DataFrame({"OR": np.exp(gee.params), "CI_low": np.exp(ci[0]), "CI_high": np.exp(ci[1]),
                              "p": gee.pvalues})
    logit_tab.to_csv(TABLES / "inheritance_logistic_chair25.csv")

    # descriptives
    desc = lk.groupby("leader_maker_cat").agg(
        graduates=("person_id", "size"), became_chair=("event", "sum"),
        median_grad_year=("graduation", "median")).assign(pct=lambda x: 100 * x.became_chair / x.graduates)
    desc.to_csv(TABLES / "inheritance_by_leader_maker_cat.csv")

    chairs = lk[lk.event == 1]
    summary = {
        "cohort_n": len(df), "with_chair_link": int(df.has_chair_link.sum()), "chairs": int(df.event.sum()),
        "female_graduates": int(df.female.sum()), "female_chairs": int(df[df.event == 1].female.sum()),
        "share_graduates_trained_by_leader_maker": round((lk.leader_makers_prior > 0).mean(), 3),
        "share_chairs_trained_by_leader_maker": round((chairs.leader_makers_prior > 0).mean(), 3),
        "median_years_grad_to_chair": float(chairs.time.median()),
        "cushing_lineage_share_by_era": df.groupby("grad_era").cushing_lineage.mean().round(3).to_dict(),
    }
    (TABLES / "inheritance_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(desc.to_string())
    print(cox_tab.round(3).to_string(index=False))
    print(logit_tab.round(3).to_string())


if __name__ == "__main__":
    main()
