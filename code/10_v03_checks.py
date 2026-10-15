"""Robustness checks: mentor concentration, women versus pipeline lag sensitivity, early-era ascertainment.

K1  Mentor concentration vs a size-only null. For each era of first chair
    appointment: observed share of new chairs trained by the era's 10 most
    prolific mentors, vs the share expected if the same number of chairs were a
    random draw of graduates from the same graduation window (so mentors with
    more trainees get proportionally more chairs). Also Gini across mentors.
K2  Women vs pipeline: sensitivity to the graduation-to-chair lag (fixed 8/12/16/
    20 y and an era-specific lag equal to the era's median years post-graduation).
K3  Early-era ascertainment: chair rate among recorded graduates by decade.

Run:  python code/10_v03_checks.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
RNG = np.random.default_rng(3)
N_SIM = 2000


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


lc = load("lc", "08_chair_lifecycle.py")
rob = load("rob", "06_robustness.py")
persons = lc.persons
positions = lc.positions
MENTORS = lc.MENTORS


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


ERAS5 = [("<1950", 0, 1949), ("1950-1969", 1950, 1969), ("1970-1989", 1970, 1989),
         ("1990-2009", 1990, 2009), ("2010-2026", 2010, 2100)]
ERAS4 = [("pre-1970", 0, 1969), ("1970-1989", 1970, 1989), ("1990-2009", 1990, 2009), ("2010-present", 2010, 2100)]


def k1(eras=ERAS5, suffix=""):
    first = positions[positions.designation == "chair"].groupby("person_id").start.min()
    grads = persons.dropna(subset=["graduation"])[["person_id", "graduation"]]
    m = MENTORS.merge(grads, on="person_id")  # trainee, mentor, graduation
    # a trainee can have two mentors (chair and PD at graduation): both are kept as rows,
    # and each new chair is counted once per mentor, as in script 08
    m["is_chair"] = m.person_id.isin(first.index)
    m["chair_year"] = m.person_id.map(first)
    rows = []
    for era, lo, hi in eras:
        ch = m[m.is_chair & m.chair_year.between(lo, hi)]
        chairs = ch.person_id.unique()
        n = len(chairs)
        g_lo, g_hi = ch.groupby("person_id").graduation.first().quantile([0.1, 0.9])
        pool_people = m[m.graduation.between(g_lo, g_hi)].person_id.unique()
        by_person = m[m.person_id.isin(pool_people)].groupby("person_id").mentor.agg(list)

        def top10_share_and_gini(people):
            cnt = {}
            for p in people:
                for mm in set(by_person.get(p, [])):
                    cnt[mm] = cnt.get(mm, 0) + 1
            c = np.array(sorted(cnt.values(), reverse=True), float)
            # share of chairs (people) trained by top-10 mentors
            top = set(sorted(cnt, key=cnt.get, reverse=True)[:10])
            share = np.mean([bool(set(by_person.get(p, [])) & top) for p in people])
            return share, gini(c), len(cnt)

        obs_share, obs_gini, n_mentors = top10_share_and_gini(chairs)
        sims = [top10_share_and_gini(RNG.choice(pool_people, n, replace=False)) for _ in range(N_SIM)]
        s_share = np.array([s[0] for s in sims]); s_gini = np.array([s[1] for s in sims])
        rows.append({
            "era": era, "new_chairs": n, "mentors_of_chairs": n_mentors,
            "graduation_window": f"{g_lo:.0f}-{g_hi:.0f}", "graduates_in_pool": len(pool_people),
            "top10_share_observed": round(obs_share, 3), "top10_share_null": round(s_share.mean(), 3),
            "top10_null_p05": round(np.percentile(s_share, 5), 3), "top10_null_p95": round(np.percentile(s_share, 95), 3),
            "top10_p": rob.pval_high(obs_share, s_share),
            "gini_observed": round(obs_gini, 3), "gini_null": round(s_gini.mean(), 3),
            "gini_p": rob.pval_high(obs_gini, s_gini),
        })
    out = pd.DataFrame(rows)
    out["ratio_top10"] = (out.top10_share_observed / out.top10_share_null).round(2)
    out.to_csv(TABLES / f"v03_mentor_concentration_vs_null{suffix}.csv", index=False)
    return out


def k2():
    grads = rob.grads_na
    share = grads.assign(f=(grads.gender == "female").astype(float)).groupby("graduation").f.agg(["sum", "size"])
    years = np.arange(share.index.min(), share.index.max() + 1)
    share = share.reindex(years, fill_value=0)
    roll = (share["sum"].rolling(5, center=True, min_periods=1).sum()
            / share["size"].rolling(5, center=True, min_periods=1).sum())
    e = rob.edges.dropna(subset=["start"]).copy()
    era_lag = {"pre-1970": 9, "1970-1989": 10.5, "1990-2009": 13.5, "2010-present": 16}  # era medians of years after graduation (Table 1)
    rows = []
    for name, lag in [("8", 8), ("12", 12), ("16", 16), ("20", 20), ("era-specific", None)]:
        d = e.copy()
        d["lag"] = lag if lag is not None else d.era.map(era_lag)
        d["p"] = (d.start - d.lag).round().map(roll)
        d = d.dropna(subset=["p"])
        for lab, g in [("1990-2009", d[d.era == "1990-2009"]), ("2010-present", d[d.era == "2010-present"]),
                       ("1990-present", d[d.start >= 1990])]:
            p = g.p.to_numpy(); obs = int((g.gender == "female").sum())
            sims = RNG.binomial(1, np.tile(p, (N_SIM, 1))).sum(1)
            rows.append({"lag_years": name, "era": lab, "appointments": len(g), "women_observed": obs,
                         "women_expected": round(float(p.sum()), 1),
                         "ratio": round(obs / p.sum(), 2) if p.sum() else None,
                         "p_lower": rob.pval_low(obs, sims)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "v03_women_pipeline_lag_sensitivity.csv", index=False)
    return out


def k3():
    chairs = set(positions.loc[positions.designation == "chair", "person_id"])
    g = persons[persons.residency_id.isin(lc.NA) & persons.graduation.notna()].copy()
    g["decade"] = (g.graduation // 10 * 10).astype(int)
    g["chair"] = g.person_id.isin(chairs)
    out = g.groupby("decade").agg(graduates=("person_id", "size"), chairs=("chair", "sum"))
    out["chair_pct"] = (100 * out.chairs / out.graduates).round(1)
    out = out[(out.index >= 1900) & (out.index <= 2010)]
    out.to_csv(TABLES / "v03_chair_rate_by_graduation_decade.csv")
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print("K1 mentor concentration vs size-only null\n", k1().to_string(index=False))
    print("K1 (4 eras)\n", k1(ERAS4, "_4era").to_string(index=False))
    print("\nK2 women vs pipeline, lag sensitivity\n", k2().to_string(index=False))
    print("\nK3 chair rate by graduation decade\n", k3().to_string())
