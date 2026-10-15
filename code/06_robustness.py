"""Robustness analyses, run after scripts 03 and 04.

R1  Out-of-sample hierarchy: rank programs on chair placements before 1990 and
    test whether that ranking predicts the direction of placements from 1990 on,
    against permutation nulls. Addresses circularity: the primary ranking is fit to
    minimize upward moves on the same data it is evaluated on.
R2  Self-hiring vs chance: degree-preserving permutation of destinations.
R3  Concentration vs program size: is chair production more concentrated than if
    chairs came from programs in proportion to their graduates (same era)?
R4  Inheritance robustness: program x graduation-decade strata; fixed 15-year
    follow-up window by era (removes unequal follow-up); early vs late hazard
    (proportional hazards); mentor's LATER chair-producing trainees (stable
    mentor/era quality vs sponsorship accumulated before graduation).
R5  Women chairs vs pipeline: expected share given the proportion of women
    among graduates ~12 years (median graduation-to-chair lag) before appointment.

Run:  python code/06_robustness.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
RNG = np.random.default_rng(2026)
N_PERM = 2000
LAG = 12  # median years from graduation to first chair, rounded from median_years_grad_to_chair in inheritance_summary.json (script 04)


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ph = load("ph", "03_placement_hierarchy.py")
inh = load("inh", "04_leadership_inheritance.py")

inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
persons = pd.read_csv(PROC / "persons.csv")
edges = pd.read_csv(TABLES / "placement_edges_chair.csv")
edges = edges[edges.source.isin(NA) & edges.target.isin(NA)].copy()
grads_na = persons[persons.residency_id.isin(NA) & persons.graduation.notna()]


def pval_low(obs, null):
    return float((np.sum(null <= obs) + 1) / (len(null) + 1))


def pval_high(obs, null):
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def summarize(obs, null, tail):
    return {"observed": round(float(obs), 4), "null_mean": round(float(np.mean(null)), 4),
            "null_p05": round(float(np.percentile(null, 5)), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4),
            "p": pval_low(obs, null) if tail == "low" else pval_high(obs, null)}


# ------------------------------------------------------------------ R1 ----
def mvr_ranks(ext, restarts=8):
    nodes = np.array(sorted(set(ext.source) | set(ext.target)))
    A = ph.adjacency(ext.source.to_numpy(), ext.target.to_numpy(), nodes)
    fits = [ph.anneal(A, s) for s in range(restarts)]
    best = min(c for _, c in fits)
    ranks = []
    for o, c in fits:
        if c == best:
            r = np.empty(len(nodes))
            r[o] = np.arange(len(nodes))
            ranks.append(r)
    return pd.Series(np.mean(ranks, axis=0), index=nodes)


def upward_fraction(src, tgt, rank):
    """Share of placements (distinct ranks only) moving to a better-ranked program."""
    rs, rt = rank.reindex(src).to_numpy(), rank.reindex(tgt).to_numpy()
    ok = (rs != rt) & ~np.isnan(rs) & ~np.isnan(rt) & (np.asarray(src) != np.asarray(tgt))
    return float(np.mean(rs[ok] > rt[ok])), int(ok.sum())


def r1():
    ext = edges[~edges.self_hire]
    train, test = ext[ext.start < 1990], ext[ext.start >= 1990]
    r_train = mvr_ranks(train)
    t = test[test.source.isin(r_train.index) & test.target.isin(r_train.index)]
    src, tgt = t.source.to_numpy(), t.target.to_numpy()
    obs, n_used = upward_fraction(src, tgt, r_train)
    null = np.array([upward_fraction(src, RNG.permutation(tgt), r_train)[0] for _ in range(N_PERM)])

    # comparison: rank programs simply by chairs produced before 1990
    produced = train.groupby("source").size().reindex(r_train.index, fill_value=0)
    prod_rank = (-produced).rank(method="average")
    prod_obs, prod_n = upward_fraction(src, tgt, prod_rank)

    r_test = mvr_ranks(test)
    common = r_train.index.intersection(r_test.index)
    rho, rho_p = stats.spearmanr(r_train[common], r_test[common])
    top10_train = set(r_train.nsmallest(10).index)
    return {
        "train_placements_pre1990": len(train), "test_placements_1990on": len(test),
        "test_placements_between_ranked_programs": n_used,
        "upward_fraction_under_pre1990_ranking": summarize(obs, null, "low"),
        "upward_fraction_under_pre1990_production_ranking": {"observed": round(prod_obs, 4), "n": prod_n},
        "spearman_pre1990_vs_1990on_ranks": {"rho": round(float(rho), 3), "p": float(rho_p), "n_programs": len(common)},
        "share_of_1990on_chairs_trained_by_pre1990_top10": round(float(test.source.isin(top10_train).mean()), 3),
    }


# ------------------------------------------------------------------ R2 ----
def selfhire(e):
    s, t = e.source.to_numpy(), e.target.to_numpy()
    null = np.array([np.mean(s == RNG.permutation(t)) for _ in range(N_PERM)])
    return summarize(np.mean(s == t), null, "high") | {"n": len(e)}


def r2():
    out = {"all_eras": selfhire(edges)}
    for era, g in edges.groupby("era"):
        out[era] = selfhire(g)
    return out


# ------------------------------------------------------------------ R3 ----
def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


def r3():
    rows = []
    for era, sub in edges.groupby("era"):
        lo, hi = sub.graduation.quantile([0.1, 0.9])
        g = grads_na[grads_na.graduation.between(lo, hi)]
        size = g.groupby("residency_id").size()
        prod = sub.groupby("source").size().reindex(size.index, fill_value=0)
        n = int(prod.sum())  # chairs trained by programs with graduates in the window
        obs_top10 = np.sort(prod.to_numpy())[::-1][:10].sum() / n
        obs_gini = gini(prod)
        p = (size / size.sum()).to_numpy()
        sims = RNG.multinomial(n, p, size=N_PERM)
        null_top10 = np.sort(sims, axis=1)[:, ::-1][:, :10].sum(1) / n
        null_gini = np.array([gini(s) for s in sims])
        rows.append({"era": era, "chairs": n, "graduation_window": f"{lo:.0f}-{hi:.0f}",
                     "programs_with_graduates": len(size),
                     "top10_share_observed": round(obs_top10, 3), "top10_share_size_only": round(null_top10.mean(), 3),
                     "top10_p": pval_high(obs_top10, null_top10),
                     "gini_observed": round(obs_gini, 3), "gini_size_only": round(null_gini.mean(), 3),
                     "gini_p": pval_high(obs_gini, null_gini)})
    order = {e: i for i, (_, _, e) in enumerate(ph.ERAS)}
    return pd.DataFrame(rows).sort_values("era", key=lambda s: s.map(order))


# ------------------------------------------------------------------ R4 ----
def later_maker_counts(df, first_chair):
    long = pd.concat([df[["person_id", "chair_id", "graduation"]].rename(columns={"chair_id": "mentor"}),
                      df[["person_id", "pd_id", "graduation"]].rename(columns={"pd_id": "mentor"})]).dropna()
    by_mentor = {m: g[["person_id", "graduation"]].drop_duplicates() for m, g in long.groupby("mentor")}
    out = {}
    for r in df[["person_id", "chair_id", "pd_id", "graduation"]].itertuples(index=False):
        ms = [m for m in (r.chair_id, r.pd_id) if isinstance(m, str)]
        if not ms:
            continue
        pool = pd.concat([by_mentor[m] for m in ms]).drop_duplicates("person_id")
        later = pool[(pool.graduation > r.graduation) & (pool.person_id != r.person_id)]
        out[r.person_id] = int(later.person_id.map(first_chair).notna().sum())
    return pd.Series(out)


def with_sex(sub, covars):
    return covars if sub.loc[sub.event == 1, "female"].sum() > 0 else [c for c in covars if c != "female"]


def pick(res, terms):
    return res[res.term.isin(terms)]


def r4():
    positions = pd.read_csv(PROC / "positions.csv")
    first_chair = positions[positions.designation == "chair"].groupby("person_id").start.min()
    cont = ["log_leader_makers_prior", "log_leader_trainees_prior", "female", "grad_year_c"]
    cat = ["maker_1_2", "maker_3p", "log_leader_trainees_prior", "female", "grad_year_c"]
    focus = ["log_leader_makers_prior", "maker_3p", "maker_1_2", "log_leader_makers_later"]

    def prep(d):
        d = d[d.has_chair_link == 1].copy()
        d["maker_1_2"] = (d.leader_maker_cat == "1-2").astype(int)
        d["maker_3p"] = (d.leader_maker_cat == "3+").astype(int)
        d["prog_decade"] = d.residency_id + "_" + (d.graduation // 10 * 10).astype(int).astype(str)
        return d

    lk = prep(pd.read_csv(PROC / "inheritance_cohort.csv"))
    res = [
        inh.cox(lk, cont, "prog_decade", "chair_id", "R4a strata program x decade"),
        inh.cox(lk, cat, "prog_decade", "chair_id", "R4a strata program x decade, categorical"),
    ]

    # fixed 15-year window: every graduate through 2010 has complete 15-y follow-up
    inh.MAX_FU = 15
    c15 = prep(inh.build_cohort())
    inh.MAX_FU = 40
    c15 = c15[c15.graduation <= 2010]
    for lab, sub in [("<1970", c15[c15.graduation < 1970]), ("1970-1989", c15[c15.graduation.between(1970, 1989)]),
                     ("1990-2010", c15[c15.graduation >= 1990])]:
        res.append(inh.cox(sub, with_sex(sub, cont), "residency_id", "chair_id", f"R4b 15-y window, grads {lab}"))
        res.append(inh.cox(sub, with_sex(sub, cat), "residency_id", "chair_id", f"R4b 15-y window categorical, grads {lab}"))

    # proportional hazards: first 12 years vs after 12 years (clock reset at 12)
    early = lk.assign(event=((lk.event == 1) & (lk.time <= 12)).astype(int), time=lk.time.clip(upper=12))
    late = lk[lk.time > 12].assign(time=lambda d: d.time - 12)
    res.append(inh.cox(early, cont, "residency_id", "chair_id", "R4c hazard within 12 y of graduation"))
    res.append(inh.cox(late, cont, "residency_id", "chair_id", "R4c hazard after 12 y"))

    # mentor's later trainees who became chairs (ever), excluding the graduate
    lk["leader_makers_later"] = lk.person_id.map(later_maker_counts(lk, first_chair)).fillna(0)
    lk["log_leader_makers_later"] = np.log1p(lk.leader_makers_later)
    res.append(inh.cox(lk, cont + ["log_leader_makers_later"], "residency_id", "chair_id",
                       "R4d prior + later chair-producing trainees"))
    for lab, sub in [("<1970", lk[lk.graduation < 1970]), ("1970+", lk[lk.graduation >= 1970])]:
        res.append(inh.cox(sub, with_sex(sub, cont) + ["log_leader_makers_later"], "residency_id", "chair_id",
                           f"R4d prior + later, grads {lab}"))
    full = pd.concat(res)
    full.to_csv(TABLES / "robustness_inheritance_models.csv", index=False)
    return pick(full, focus)


# ------------------------------------------------------------------ R5 ----
def r5():
    share = grads_na.assign(f=(grads_na.gender == "female").astype(float)).groupby("graduation").f.agg(["sum", "size"])
    years = np.arange(share.index.min(), share.index.max() + 1)
    share = share.reindex(years, fill_value=0)
    roll = (share["sum"].rolling(5, center=True, min_periods=1).sum()
            / share["size"].rolling(5, center=True, min_periods=1).sum())
    rows = []
    e = edges.dropna(subset=["start"]).copy()
    e["p_female_pipeline"] = (e.start - LAG).map(roll)
    e = e.dropna(subset=["p_female_pipeline"])
    for era, g in list(e.groupby("era")) + [("1990-present", e[e.start >= 1990])]:
        p = g.p_female_pipeline.to_numpy()
        obs = int((g.gender == "female").sum())
        sims = RNG.binomial(1, np.tile(p, (N_PERM, 1))).sum(1)
        rows.append({"era": era, "chair_appointments": len(g), "women_observed": obs,
                     "women_expected_from_pipeline": round(float(p.sum()), 1),
                     "observed_share": round(obs / len(g), 3), "pipeline_share": round(float(p.mean()), 3),
                     "ratio_observed_expected": round(obs / p.sum(), 2) if p.sum() else None,
                     "p_lower": pval_low(obs, sims)})
    return pd.DataFrame(rows)


def main():
    summary = {"R1_out_of_sample_hierarchy": r1(), "R2_self_hire_vs_null": r2()}
    r3t = r3()
    r3t.to_csv(TABLES / "robustness_concentration_vs_size.csv", index=False)
    r5t = r5()
    r5t.to_csv(TABLES / "robustness_women_pipeline.csv", index=False)
    r4t = r4()
    (TABLES / "robustness_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("\nR3 concentration vs size-only null\n", r3t.to_string(index=False))
    print("\nR4 inheritance robustness (focus terms)\n", r4t.round(3).to_string(index=False))
    print("\nR5 women vs pipeline\n", r5t.to_string(index=False))


if __name__ == "__main__":
    main()
