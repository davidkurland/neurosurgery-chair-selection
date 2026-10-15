"""Sensitivity analyses for the within-program leader-maker association: program-era
production and trend form.

In script 06 (R4a), the within-program leader-maker association is attenuated when
strata are program x graduation decade. This script tests whether program "eras"
(periods when a program produced many chairs) raise both the mentor's leader-maker
count and the trainee's own chance, rather than mentor-specific inheritance.

F1  within-program model + program's recent chair production: share of the
    program's graduates in the previous 10 graduation years who had become chairs
    before the graduate's graduation year (time-respecting)
F2  same, categorical exposure
F3  within-program model with a cubic graduation-year trend instead of linear
F4  F1 by graduation era
H1  comparison of the pre-1990 minimum-violation ranking with a simple
    chairs-produced ranking, scored on the same post-1990 placements (untied in both)

Run:  python code/07_followup_program_era.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


inh = load("inh", "04_leadership_inheritance.py")
rob = load("rob", "06_robustness.py")


def recent_program_rate(df, window=10):
    rows = {}
    for _, g in df.groupby("residency_id"):
        gy = g.graduation.to_numpy()
        fy = g.first_chair_year.to_numpy()
        for pid, y in zip(g.person_id, gy):
            m = (gy < y) & (gy >= y - window)
            n = m.sum()
            c = np.sum(m & (np.nan_to_num(fy, nan=9999) < y))
            rows[pid] = c / n if n else 0.0
    return pd.Series(rows)


def with_sex(sub, covars):
    return covars if sub.loc[sub.event == 1, "female"].sum() > 0 else [c for c in covars if c != "female"]


def inheritance():
    df = pd.read_csv(PROC / "inheritance_cohort.csv")
    df["prog_recent_rate_10"] = df.person_id.map(recent_program_rate(df)) * 10
    lk = df[df.has_chair_link == 1].copy()
    lk["maker_1_2"] = (lk.leader_maker_cat == "1-2").astype(int)
    lk["maker_3p"] = (lk.leader_maker_cat == "3+").astype(int)
    lk["gy2"] = lk.grad_year_c ** 2
    lk["gy3"] = lk.grad_year_c ** 3
    cont = ["log_leader_makers_prior", "log_leader_trainees_prior", "female", "grad_year_c"]
    cat = ["maker_1_2", "maker_3p", "log_leader_trainees_prior", "female", "grad_year_c"]
    res = [
        inh.cox(lk, cont + ["prog_recent_rate_10"], "residency_id", "chair_id", "F1 within-program + recent program rate"),
        inh.cox(lk, cat + ["prog_recent_rate_10"], "residency_id", "chair_id", "F2 categorical + recent program rate"),
        inh.cox(lk, cont + ["gy2", "gy3"], "residency_id", "chair_id", "F3 cubic graduation-year trend"),
        inh.cox(lk, cat + ["gy2", "gy3"], "residency_id", "chair_id", "F3 categorical, cubic trend"),
    ]
    for lab, sub in [("<1970", lk[lk.graduation < 1970]), ("1970+", lk[lk.graduation >= 1970])]:
        res.append(inh.cox(sub, with_sex(sub, cont) + ["prog_recent_rate_10"], "residency_id", "chair_id",
                           f"F4 within-program + recent program rate, grads {lab}"))
        res.append(inh.cox(sub, with_sex(sub, cat) + ["prog_recent_rate_10"], "residency_id", "chair_id",
                           f"F4 categorical + recent program rate, grads {lab}"))
    out = pd.concat(res)
    out.to_csv(TABLES / "followup_program_era_models.csv", index=False)
    desc = lk.groupby("leader_maker_cat").prog_recent_rate_10.describe()[["mean", "50%"]].round(2)
    corr = lk[["log_leader_makers_prior", "prog_recent_rate_10"]].corr().iloc[0, 1]
    return out, desc, corr


def hierarchy_ranking_comparison():
    ext = rob.edges[~rob.edges.self_hire]
    train, test = ext[ext.start < 1990], ext[ext.start >= 1990]
    r_mvr = rob.mvr_ranks(train)
    produced = train.groupby("source").size().reindex(r_mvr.index, fill_value=0)
    r_prod = (-produced).rank(method="average")
    t = test[test.source.isin(r_mvr.index) & test.target.isin(r_mvr.index)]
    s, g = t.source.to_numpy(), t.target.to_numpy()
    ms, mg = r_mvr.reindex(s).to_numpy(), r_mvr.reindex(g).to_numpy()
    ps, pg = r_prod.reindex(s).to_numpy(), r_prod.reindex(g).to_numpy()
    both = (ms != mg) & (ps != pg) & (s != g)
    return {
        "placements_untied_in_both": int(both.sum()),
        "upward_fraction_mvr": round(float(np.mean(ms[both] > mg[both])), 3),
        "upward_fraction_production": round(float(np.mean(ps[both] > pg[both])), 3),
        "placements_from_programs_with_no_pre1990_chairs": int((produced.reindex(s).to_numpy() == 0).sum()),
        "n_test_placements_ranked": len(t),
    }


def main():
    models, desc, corr = inheritance()
    fair = hierarchy_ranking_comparison()
    (TABLES / "followup_hierarchy_fair_comparison.json").write_text(json.dumps(fair, indent=1))
    focus = models[models.term.isin(["log_leader_makers_prior", "maker_1_2", "maker_3p", "prog_recent_rate_10"])]
    print(focus.round(3).to_string(index=False))
    print("\nrecent program rate (per 10 pp) by leader-maker category\n", desc)
    print("corr(log leader-makers, recent program rate):", round(corr, 3))
    print("\nH1 ranking comparison:", json.dumps(fair, indent=1))


if __name__ == "__main__":
    main()
