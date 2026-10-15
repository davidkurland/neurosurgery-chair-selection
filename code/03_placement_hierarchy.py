"""Placement hierarchy of neurosurgery residency programs.

Directed, weighted placement network: edge u -> v when a graduate of program u
holds a leadership position (chair; or chair/PD) at program v. One edge per
(person, role, host program) regardless of how many terms.

Analyses
  * Concentration of leadership production (Gini, top-k shares), overall and by era
  * Self-hire ("inbreeding") rate by era
  * Minimum violation ranking (MVR; Clauset, Arbesman & Larremore, Sci Adv 2015):
    ordering of programs that minimises "upward" placements, via simulated
    annealing; rank uncertainty from edge bootstrap
  * Hierarchy strength: observed violation fraction vs degree-preserving nulls
  * Placement step sizes (how far down/up the hierarchy leaders move) and distance

Run:  python code/03_placement_hierarchy.py [--boot 200] [--nulls 50] [--workers 8] [--roles chair,leader]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

ERAS = [(0, 1969, "pre-1970"), (1970, 1989, "1970-1989"), (1990, 2009, "1990-2009"), (2010, 9999, "2010-present")]


def era_of(year):
    for lo, hi, label in ERAS:
        if lo <= year <= hi:
            return label
    return None


# ---------------------------------------------------------------- network ----
def placement_edges(role):
    persons = pd.read_csv(PROC / "persons.csv")
    positions = pd.read_csv(PROC / "positions.csv")
    inst = pd.read_csv(PROC / "institutions.csv")
    # Unit = person x host program. "chair": chair terms only. "leader": chair or PD,
    # so chair+PD at the same program is ONE edge, PD at X then chair at Y is two.
    # "*_first": sensitivity keeping only each person's first appointment.
    base = role.removesuffix("_first")
    pos = positions if base == "leader" else positions[positions.designation == base]
    e = (pos.merge(persons[["person_id", "name", "gender", "graduation", "residency_id"]], on="person_id")
            .dropna(subset=["residency_id", "institution_id"])
            .sort_values("start")
            .groupby(["person_id", "institution_id"], as_index=False).first())
    if role.endswith("_first"):
        e = e.sort_values("start").groupby("person_id", as_index=False).first()
    e = e.rename(columns={"residency_id": "source", "institution_id": "target"})
    e["self_hire"] = e.source == e.target
    e["era"] = e.start.map(lambda y: era_of(y) if pd.notna(y) else None)
    loc = inst.set_index("institution_id")
    for end in ("source", "target"):
        e[f"{end}_name"] = e[end].map(loc.institution)
        e[f"{end}_lat"] = e[end].map(loc.lat)
        e[f"{end}_lon"] = e[end].map(loc.lon)
        e[f"{end}_state"] = e[end].map(loc.location).str.extract(r",\s*([A-Z]{2})$")[0]
    e["distance_km"] = haversine(e.source_lat, e.source_lon, e.target_lat, e.target_lon)
    e["same_state"] = e.source_state.notna() & (e.source_state == e.target_state)
    return e


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(x.astype(float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return np.nan
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


# -------------------------------------------------------------------- MVR ----
def violations(A, order):
    P = A[np.ix_(order, order)]
    return np.tril(P, -1).sum()  # row lower in ranking placing into a higher-ranked column


def anneal(A, seed, sweeps=400):
    """Simulated annealing over orderings; position 0 = top of hierarchy."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    score = A.sum(1) - A.sum(0) + rng.normal(0, 1e-3, n)  # net exporters first
    order = list(np.argsort(-score))
    cost = violations(A, order)
    best, best_cost = order[:], cost
    steps = sweeps * n
    temps = np.geomspace(2.0, 0.01, steps)
    for t in temps:
        a, b = sorted(rng.choice(n, 2, replace=False))
        x, y = order[a], order[b]
        mid = order[a + 1:b]
        delta = A[x, y] - A[y, x]
        if mid:
            delta += A[mid, y].sum() + A[x, mid].sum() - A[mid, x].sum() - A[y, mid].sum()
        if delta <= 0 or rng.random() < np.exp(-delta / t):
            order[a], order[b] = y, x
            cost += delta
            if cost < best_cost:
                best, best_cost = order[:], cost
    return np.array(best), best_cost


def adjacency(src, tgt, nodes):
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s, t in zip(src, tgt):
        if s != t:
            A[idx[s], idx[t]] += 1
    return A


def _boot_job(args):
    """Cluster bootstrap over people: a chair who led several programs contributes
    several edges, so resample people (with all their edges), not edges."""
    src, tgt, pid, nodes, seed = args
    rng = np.random.default_rng(seed)
    people = np.unique(pid)
    drawn = rng.choice(people, len(people), replace=True)
    by_person = {p: np.flatnonzero(pid == p) for p in people}
    i = np.concatenate([by_person[p] for p in drawn])
    A = adjacency(src[i], tgt[i], nodes)
    order, _ = anneal(A, seed)
    ranks = np.empty(len(nodes))
    ranks[order] = np.arange(len(nodes))
    return ranks


def _null_job(args):
    src, tgt, nodes, seed = args
    rng = np.random.default_rng(seed)
    t = tgt.copy()
    rng.shuffle(t)  # preserves every program's out- and in-degree
    A = adjacency(src, t, nodes)
    _, cost = anneal(A, seed)
    return cost / A.sum()


# ------------------------------------------------------------------- main ----
def run(role, n_boot, n_null, workers):
    e = placement_edges(role)
    e.to_csv(TABLES / f"placement_edges_{role}.csv", index=False)
    persons = pd.read_csv(PROC / "persons.csv")
    inst = pd.read_csv(PROC / "institutions.csv").set_index("institution_id")

    # Rank a closed system: North American programs only (the database is US-centric,
    # and one-off European sources otherwise float to the top on a single placement)
    na = set(inst.index[inst.country.isin(["United States", "Canada"])])
    trained_outside_na = int((~e.source.isin(na)).sum())
    e = e[e.source.isin(na) & e.target.isin(na)]
    ext = e[~e.self_hire]
    nodes = np.array(sorted(set(ext.source) | set(ext.target)))
    src, tgt = ext.source.to_numpy(), ext.target.to_numpy()
    A = adjacency(src, tgt, nodes)

    orders = []
    with ProcessPoolExecutor(workers) as ex:
        fits = list(ex.map(anneal, [A] * 8, range(8)))
    best_cost = min(c for _, c in fits)
    for o, c in fits:
        if c == best_cost:
            r = np.empty(len(nodes)); r[o] = np.arange(len(nodes)); orders.append(r)
    point_rank = np.mean(orders, axis=0)
    rho_obs = best_cost / A.sum()

    with ProcessPoolExecutor(workers) as ex:
        boots = np.array(list(ex.map(_boot_job, [(src, tgt, ext.person_id.to_numpy(), nodes, 1000 + b) for b in range(n_boot)])))
        nulls = np.array(list(ex.map(_null_job, [(src, tgt, nodes, 5000 + b) for b in range(n_null)])))

    grads = persons.dropna(subset=["graduation"]).groupby("residency_id").size()
    grads = grads[grads.index.isin(na)]
    first_grad = persons.groupby("residency_id").graduation.min()
    out_all = e.groupby("source").size()
    self_ = e[e.self_hire].groupby("source").size()
    hired = e.groupby("target").size()
    n = len(nodes)
    rank = pd.DataFrame({
        "institution_id": nodes,
        "program": [inst.institution.get(v) for v in nodes],
        "location": [inst.location.get(v) for v in nodes],
        "mvr_rank": point_rank + 1,
        "mvr_pct_top": 100 * (1 - point_rank / (n - 1)),
        "boot_rank_median": np.median(boots, axis=0) + 1,
        "boot_rank_p05": np.percentile(boots, 5, axis=0) + 1,
        "boot_rank_p95": np.percentile(boots, 95, axis=0) + 1,
        "leaders_produced": [out_all.get(v, 0) for v in nodes],
        "leaders_produced_elsewhere": A.sum(1),
        "self_hires": [self_.get(v, 0) for v in nodes],
        "leaders_hired": [hired.get(v, 0) for v in nodes],
        "graduates_listed": [grads.get(v, 0) for v in nodes],
        "first_graduate_year": [first_grad.get(v) for v in nodes],
    }).sort_values("mvr_rank")
    rank["leaders_per_100_grads"] = 100 * rank.leaders_produced / rank.graduates_listed.replace(0, np.nan)
    rank.to_csv(TABLES / f"program_ranking_{role}.csv", index=False)

    pct = dict(zip(rank.institution_id, rank.mvr_pct_top))
    ext = ext.assign(source_pct=ext.source.map(pct), target_pct=ext.target.map(pct))
    ext["step_pct"] = ext.target_pct - ext.source_pct  # negative = moved down
    ext.to_csv(TABLES / f"placement_steps_{role}.csv", index=False)

    # production concentration among all programs with graduates listed
    def conc(sub_edges, base_index):
        p = pd.Series(0, index=base_index).add(sub_edges.groupby("source").size(), fill_value=0)
        p = p.sort_values(ascending=False)
        tot = p.sum()
        return {
            "n_positions": int(tot),
            "n_producing_programs": int((p > 0).sum()),
            "gini_production": round(gini(p), 3),
            "top5_share": round(p.head(5).sum() / tot, 3) if tot else None,
            "top10_share": round(p.head(10).sum() / tot, 3) if tot else None,
            "top20pct_programs_share": round(p.head(max(1, int(0.2 * len(p)))).sum() / tot, 3) if tot else None,
            "self_hire_rate": round(sub_edges.self_hire.mean(), 3) if len(sub_edges) else None,
            "same_state_rate_external": round(sub_edges.loc[~sub_edges.self_hire, "same_state"].mean(), 3) if len(sub_edges) else None,
            "median_distance_km_external": round(sub_edges.loc[~sub_edges.self_hire, "distance_km"].median(), 0) if len(sub_edges) else None,
            "female_share": round((sub_edges.gender == "female").mean(), 3) if len(sub_edges) else None,
        }
    eras = []
    for _, _, label in ERAS:
        sub = e[e.era == label]
        s_ext = ext[ext.era == label]
        eras.append({"era": label, **conc(sub, grads.index),
                     "violation_fraction_under_overall_ranking": round((s_ext.step_pct > 0).mean(), 3) if len(s_ext) else None,
                     "median_step_pct": round(s_ext.step_pct.median(), 1) if len(s_ext) else None})
    pd.DataFrame(eras).to_csv(TABLES / f"hierarchy_by_era_{role}.csv", index=False)

    summary = {
        "role": role,
        "overall": conc(e, grads.index),
        "n_programs_in_ranking": int(n),
        "leaders_trained_outside_north_america_excluded": trained_outside_na,
        "n_external_placements": int(A.sum()),
        "mvr_violation_fraction": round(rho_obs, 3),
        "null_violation_fraction_mean": round(nulls.mean(), 3),
        "null_violation_fraction_p05_p95": [round(np.percentile(nulls, 5), 3), round(np.percentile(nulls, 95), 3)],
        "hierarchy_p_value": float((np.sum(nulls <= rho_obs) + 1) / (len(nulls) + 1)),
        "share_moving_down": round((ext.step_pct < 0).mean(), 3),
        "share_moving_up_gt20pct": round((ext.step_pct > 20).mean(), 3),
        "median_step_pct": round(ext.step_pct.median(), 1),
        "n_equally_optimal_orderings_found": len(orders),
        "n_boot": n_boot, "n_null": n_null,
    }
    (TABLES / f"hierarchy_summary_{role}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(rank.head(20)[["program", "mvr_rank", "boot_rank_p05", "boot_rank_p95",
                         "leaders_produced", "self_hires", "graduates_listed"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--nulls", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--roles", default="chair,leader")
    a = ap.parse_args()
    for role in a.roles.split(","):
        run(role, a.boot, a.nulls, a.workers)
