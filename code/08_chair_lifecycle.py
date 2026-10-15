"""The changing path to the chair: tenure, age at appointment, succession and chairmaker analyses.

Chair appointments at US and Canadian programs.

L1  Tenure. Consecutive terms by the same person at the same program (gap <= 1 y)
    are merged. Terms with no end date end at the successor's start, then at the
    chair's death; otherwise censored at 2026. Kaplan-Meier median and 10-year
    restricted mean tenure by era of appointment; Cox HR per decade of appointment.
L2  Age and years since residency graduation at first chair appointment, by era;
    linear trend per decade.
L3  Succession. For consecutive chairs at a program: was the new chair a graduate
    of that program, previously its PD, or trained by the outgoing chair (chair or
    PD at graduation)? Chance expectation for "trained by the outgoing chair" =
    share of eligible neurosurgeons (graduated 5-25 y before the appointment)
    trained by the outgoing chair: nationally, and among the program's own
    graduates for own-graduate successors.
L4  Chairmakers. Chairs trained per mentor (chair or PD at graduation), second-
    generation chair descendants, and concentration of new chairs among mentors
    by era.

Run:  python code/08_chair_lifecycle.py
"""

import importlib.util
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.duration.survfunc import SurvfuncRight

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
SNAPSHOT_YEAR = 2026
RNG = np.random.default_rng(2026)
ERA_BINS = [0, 1949, 1969, 1989, 2009, 9999]
ERA_LABELS = ["<1950", "1950-1969", "1970-1989", "1990-2009", "2010-2026"]

spec = importlib.util.spec_from_file_location("inh", Path(__file__).resolve().parent / "04_leadership_inheritance.py")
inh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inh)

inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
persons = pd.read_csv(PROC / "persons.csv")
positions = pd.read_csv(PROC / "positions.csv")
links = pd.read_csv(PROC / "training_links.csv")
pinfo = persons.set_index("person_id")


def era(years):
    return pd.cut(years, ERA_BINS, labels=ERA_LABELS)


def mentor_table():
    """person -> mentor (chair or PD at graduation; derived, else site-assigned)."""
    l = links.copy()
    l["chair"] = l.derived_chair_id.fillna(l.site_chair_id)
    l["pd"] = l.derived_pd_id.fillna(l.site_pd_id)
    m = pd.concat([l[["person_id", "chair"]].rename(columns={"chair": "mentor"}),
                   l[["person_id", "pd"]].rename(columns={"pd": "mentor"})]).dropna().drop_duplicates()
    return m[m.person_id != m.mentor]


MENTORS = mentor_table()
MENTORS_OF = MENTORS.groupby("person_id").mentor.agg(set).to_dict()


# ------------------------------------------------------------------ L1 ----
def tenures():
    ch = positions[(positions.designation == "chair") & positions.institution_id.isin(NA)]
    rows = []
    for (pid, iid), g in ch.sort_values("start").groupby(["person_id", "institution_id"]):
        cur = None
        for r in g.itertuples():
            if cur is not None and pd.notna(cur["end"]) and r.start - cur["end"] <= 1:
                cur["end"] = max(cur["end"], r.end) if pd.notna(r.end) else np.nan
            else:
                if cur is not None:
                    rows.append(cur)
                cur = {"person_id": pid, "institution_id": iid, "start": r.start, "end": r.end}
        rows.append(cur)
    t = pd.DataFrame(rows).sort_values(["institution_id", "start"]).reset_index(drop=True)

    next_start = []
    for r in t.itertuples():
        later = t[(t.institution_id == r.institution_id) & (t.person_id != r.person_id) & (t.start > r.start)]
        next_start.append(later.start.min() if len(later) else np.nan)
    t["next_start"] = next_start
    t["death"] = t.person_id.map(pinfo.death)
    t["event"] = 1
    t["end_source"] = np.where(t.end.notna(), "recorded", "")
    fill_next = t.end.isna() & t.next_start.notna()
    t.loc[fill_next, "end"] = t.loc[fill_next, "next_start"]
    t.loc[fill_next, "end_source"] = "successor start"
    fill_death = t.end.isna() & t.death.notna()
    t.loc[fill_death, "end"] = t.loc[fill_death, "death"]
    t.loc[fill_death, "end_source"] = "death"
    ongoing = t.end.isna()
    t.loc[ongoing, "end"] = SNAPSHOT_YEAR
    t.loc[ongoing, ["event", "end_source"]] = [0, "censored 2026"]
    t["years"] = (t.end - t.start).clip(lower=0.5)
    t["era"] = era(t.start)
    return t


def km_summary(g, horizon=10):
    sf = SurvfuncRight(g.years.to_numpy(float), g.event.to_numpy(int))
    times, surv = sf.surv_times, sf.surv_prob
    grid = np.concatenate([[0], times[times < horizon], [horizon]])
    s_left = np.concatenate([[1], surv[times < horizon]])
    rmst = float(np.sum(np.diff(grid) * s_left))
    below = np.flatnonzero(surv <= 0.5)
    median = float(times[below[0]]) if len(below) else np.nan
    return median, rmst, sf


def l1():
    t = tenures()
    t.to_csv(TABLES / "lifecycle_tenures.csv", index=False)
    rows, curves = [], []
    for e, g in t.groupby("era", observed=True):
        median, rmst, sf = km_summary(g)
        rows.append({"era": e, "tenures": len(g), "ongoing": int((g.event == 0).sum()),
                     "km_median_years": median, "rmst_10y": round(rmst, 2),
                     "pct_serving_10y": round(100 * float(np.interp(10, sf.surv_times, sf.surv_prob, left=1)), 1)})
        curves.append(pd.DataFrame({"era": e, "years": np.concatenate([[0], sf.surv_times]),
                                    "surv": np.concatenate([[1], sf.surv_prob])}))
    by_era = pd.DataFrame(rows)
    by_era.to_csv(TABLES / "lifecycle_tenure_by_era.csv", index=False)
    pd.concat(curves).to_csv(TABLES / "lifecycle_tenure_km_curves.csv", index=False)
    t["decade_c"] = (t.start - 1990) / 10
    t["all"] = "all"
    t["time"] = t.years
    cox = inh.cox(t, ["decade_c"], "all", "institution_id", "tenure ~ decade of appointment")
    return by_era, cox, t


# ------------------------------------------------------------------ L2 ----
def l2(t):
    first = t.sort_values("start").groupby("person_id", as_index=False).first()
    first["birth"] = first.person_id.map(pinfo.birth)
    first["graduation"] = first.person_id.map(pinfo.graduation)
    first["age"] = first.start - first.birth
    first["years_post_grad"] = first.start - first.graduation
    first = first[(first.years_post_grad.isna()) | (first.years_post_grad >= 0)]
    q = lambda s, p: s.quantile(p)
    by_era = first.groupby("era", observed=True).agg(
        first_appointments=("person_id", "size"),
        age_n=("age", "count"), age_median=("age", "median"),
        age_q1=("age", lambda s: q(s, .25)), age_q3=("age", lambda s: q(s, .75)),
        post_grad_median=("years_post_grad", "median"),
        post_grad_q1=("years_post_grad", lambda s: q(s, .25)), post_grad_q3=("years_post_grad", lambda s: q(s, .75)))
    by_era.to_csv(TABLES / "lifecycle_age_by_era.csv")
    first["decade_c"] = (first.start - 1990) / 10
    trend = {}
    for y in ("age", "years_post_grad"):
        fit = smf.ols(f"{y} ~ decade_c", data=first.dropna(subset=[y])).fit()
        lo, hi = fit.conf_int().loc["decade_c"]
        trend[y] = {"per_decade": round(fit.params["decade_c"], 2), "ci": [round(lo, 2), round(hi, 2)], "n": int(fit.nobs)}
    first["bin5"] = (first.start // 5 * 5).astype(int)
    first[["person_id", "start", "bin5", "age", "years_post_grad"]].to_csv(TABLES / "lifecycle_first_chair_points.csv", index=False)
    return by_era, trend


# ------------------------------------------------------------------ L3 ----
def l3(t):
    grads = persons[persons.residency_id.isin(NA) & persons.graduation.notna()][["person_id", "graduation", "residency_id"]]
    gy_sorted = np.sort(grads.graduation.to_numpy())
    trainees = MENTORS.merge(grads, on="person_id").groupby("mentor")
    trainee_idx = {m: g for m, g in trainees}
    prog_grads = {p: np.sort(g.graduation.to_numpy()) for p, g in grads.groupby("residency_id")}
    pd_pos = positions[positions.designation == "pd"]

    rows = []
    for iid, g in t.sort_values("start").groupby("institution_id"):
        g = g.reset_index(drop=True)
        for k in range(1, len(g)):
            pred, succ = g.iloc[k - 1], g.iloc[k]
            if pred.person_id == succ.person_id:
                continue
            s = succ.start
            lo, hi = s - 25, s - 5
            pool = np.searchsorted(gy_sorted, hi, "right") - np.searchsorted(gy_sorted, lo, "left")
            tr = trainee_idx.get(pred.person_id)
            tr_win = tr[tr.graduation.between(lo, hi)] if tr is not None else None
            n_tr = 0 if tr_win is None else len(tr_win)
            own_arr = prog_grads.get(iid, np.array([]))
            own_pool = np.searchsorted(own_arr, hi, "right") - np.searchsorted(own_arr, lo, "left")
            n_tr_own = 0 if tr_win is None else int((tr_win.residency_id == iid).sum())
            rows.append({
                "institution_id": iid, "start": s, "predecessor": pred.person_id, "successor": succ.person_id,
                "own_graduate": pinfo.residency_id.get(succ.person_id) == iid,
                "prior_pd_here": bool(((pd_pos.person_id == succ.person_id) & (pd_pos.institution_id == iid)
                                       & (pd_pos.start < s)).any()),
                "trained_by_predecessor": pred.person_id in MENTORS_OF.get(succ.person_id, set()),
                "p_chance_national": n_tr / pool if pool else np.nan,
                "p_chance_own": n_tr_own / own_pool if own_pool else np.nan,
            })
    s = pd.DataFrame(rows)
    s["era"] = era(s.start)
    s.to_csv(TABLES / "lifecycle_successions.csv", index=False)

    def tail(obs, p):
        p = p[~np.isnan(p)]
        sims = RNG.binomial(1, np.tile(p, (5000, 1))).sum(1)
        return float((np.sum(sims >= obs) + 1) / 5001)

    out = []
    groups = list(s.groupby("era", observed=True)) + [("All", s)]
    for e, g in groups:
        own = g[g.own_graduate]
        out.append({
            "era": e, "transitions": len(g),
            "own_graduate_pct": round(100 * g.own_graduate.mean(), 1),
            "prior_pd_here_pct": round(100 * g.prior_pd_here.mean(), 1),
            "trained_by_predecessor_n": int(g.trained_by_predecessor.sum()),
            "trained_by_predecessor_pct": round(100 * g.trained_by_predecessor.mean(), 1),
            "chance_national_expected_n": round(float(np.nansum(g.p_chance_national)), 1),
            "ratio_vs_national_chance": round(g.trained_by_predecessor.sum() / np.nansum(g.p_chance_national), 1),
            "p_vs_national": tail(int(g.trained_by_predecessor.sum()), g.p_chance_national.to_numpy()),
            "own_grad_successors": len(own),
            "own_grad_trained_by_predecessor_pct": round(100 * own.trained_by_predecessor.mean(), 1) if len(own) else np.nan,
            "own_grad_chance_pct": round(100 * np.nanmean(own.p_chance_own), 1) if len(own) else np.nan,
            "p_vs_own_program_chance": tail(int(own.trained_by_predecessor.sum()), own.p_chance_own.to_numpy()) if len(own) else np.nan,
        })
    res = pd.DataFrame(out)
    res.to_csv(TABLES / "lifecycle_succession_by_era.csv", index=False)
    return res


# ------------------------------------------------------------------ L4 ----
def l4():
    chairs_any = set(positions.loc[positions.designation == "chair", "person_id"])
    G = nx.DiGraph()
    G.add_edges_from(zip(MENTORS.mentor, MENTORS.person_id))
    grad = pinfo.graduation
    rows = []
    for m, g in MENTORS.groupby("mentor"):
        direct = set(g.person_id) & chairs_any
        second = set(MENTORS[MENTORS.mentor.isin(direct)].person_id) & chairs_any - direct - {m}
        rows.append({"mentor_id": m, "name": pinfo.name.get(m), "trainees": g.person_id.nunique(),
                     "chairs_trained": len(direct), "second_generation_chairs": len(second),
                     "first_trainee_grad": grad.reindex(list(g.person_id)).min(),
                     "last_trainee_grad": grad.reindex(list(g.person_id)).max()})
    km = pd.DataFrame(rows).sort_values(["chairs_trained", "second_generation_chairs"], ascending=False)
    top = km.head(40).copy()
    top["all_descendant_chairs"] = [len(nx.descendants(G, m) & chairs_any) for m in top.mentor_id]
    km = km.merge(top[["mentor_id", "all_descendant_chairs"]], on="mentor_id", how="left")
    km.to_csv(TABLES / "lifecycle_chairmakers.csv", index=False)

    # concentration of new chairs among mentors, by era of first chair appointment
    first = positions[positions.designation == "chair"].groupby("person_id").start.min()
    fm = MENTORS[MENTORS.person_id.isin(first.index)].assign(era=lambda d: era(d.person_id.map(first)))
    conc = []
    for e, g in fm.groupby("era", observed=True):
        counts = g.groupby("mentor").person_id.nunique().sort_values(ascending=False)
        top10 = set(counts.head(10).index)
        chairs = g.person_id.nunique()
        conc.append({"era": e, "new_chairs_with_mentor": chairs, "mentors": len(counts),
                     "share_trained_by_era_top10_mentors": round(g[g.mentor.isin(top10)].person_id.nunique() / chairs, 3),
                     "max_chairs_one_mentor": int(counts.iloc[0]), "top_mentor": pinfo.name.get(counts.index[0])})
    conc = pd.DataFrame(conc)
    conc.to_csv(TABLES / "lifecycle_chairmaker_concentration_by_era.csv", index=False)

    # two-generation tree for the top chairmakers
    edges = []
    for m in km.head(6).mentor_id:
        direct = set(MENTORS[MENTORS.mentor == m].person_id) & chairs_any
        for d in direct:
            edges.append({"root": m, "parent": m, "child": d, "generation": 1})
            for c in set(MENTORS[MENTORS.mentor == d].person_id) & chairs_any:
                edges.append({"root": m, "parent": d, "child": c, "generation": 2})
    te = pd.DataFrame(edges)
    for col in ("root", "parent", "child"):
        te[f"{col}_name"] = te[col].map(pinfo.name)
    te["child_first_chair"] = te.child.map(first)
    te.to_csv(TABLES / "lifecycle_chairmaker_tree_edges.csv", index=False)
    return km, conc


def main():
    ten, cox, t = l1()
    age, trend = l2(t)
    succ = l3(t)
    km, conc = l4()
    summary = {
        "tenure_cox_hr_per_decade_of_appointment": cox.round(3).to_dict("records"),
        "tenure_end_sources": t.end_source.value_counts().to_dict(),
        "age_trends": trend,
    }
    (TABLES / "lifecycle_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pd.set_option("display.width", 250)
    print("TENURE\n", ten.to_string(index=False))
    print(json.dumps(summary, indent=1, default=str))
    print("\nAGE\n", age.round(1).to_string())
    print("\nSUCCESSION\n", succ.to_string(index=False))
    print("\nCHAIRMAKERS\n", km.head(15)[["name", "trainees", "chairs_trained", "second_generation_chairs",
                                          "all_descendant_chairs", "first_trainee_grad", "last_trainee_grad"]].to_string(index=False))
    print("\nCHAIRMAKER CONCENTRATION\n", conc.to_string(index=False))


if __name__ == "__main__":
    main()
