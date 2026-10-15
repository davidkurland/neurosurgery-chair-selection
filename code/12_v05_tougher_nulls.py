"""Self-hiring against four nulls, plus a sensitivity analysis restricted to successions
from 1970 on (better ascertainment). Nulls, weakest to strongest:
  size_all      program's share of ALL North American graduates 5-25 y earlier (as in script 11)
  size_chairs   program's share of those graduates who ever became a chair
  size_chairs10 same, restricted to chairs first appointed within +/-10 y
  permutation   degree-preserving shuffle of successors' training programs across
                successions (preserves how many successors each program trained AND
                how many successions each program had)

Run:  python code/12_v05_tougher_nulls.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
RNG = np.random.default_rng(12)
N_SIM = 5000

s = pd.read_csv(TABLES / "lifecycle_successions.csv")
inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
per = pd.read_csv(PROC / "persons.csv")
pos = pd.read_csv(PROC / "positions.csv")

chair_pos = pos[(pos.designation == "chair") & pos.institution_id.isin(NA)]
first_chair = chair_pos.groupby("person_id").start.min()
ever_chair = set(chair_pos.person_id)

g = per[per.residency_id.isin(NA) & per.graduation.notna()].copy()
gy, prog = g.graduation.to_numpy(), g.residency_id.to_numpy()
ec = g.person_id.isin(ever_chair).to_numpy()
fc = g.person_id.map(first_chair).to_numpy(dtype=float)

s = s.merge(per[["person_id", "residency_id"]].rename(
    columns={"person_id": "successor", "residency_id": "successor_residency"}),
    on="successor", how="left")


def expectations(kind):
    out = []
    for r in s.itertuples():
        w = (gy >= r.start - 25) & (gy <= r.start - 5)
        if kind == "size_all":
            m = w
        elif kind == "size_chairs":
            m = w & ec
        else:
            m = w & ec & (np.abs(np.nan_to_num(fc, nan=9e9) - r.start) <= 10)
        out.append((prog[m] == r.institution_id).mean() if m.sum() else np.nan)
    return np.array(out)


def binom_null(d, p, label, era):
    obs = int(d.own_graduate.sum())
    sims = RNG.binomial(1, np.tile(p, (N_SIM, 1))).sum(1)
    return {"null": label, "era": era, "successions": len(d), "observed_n": obs,
            "observed_pct": round(100 * obs / len(d), 1),
            "expected_pct": round(100 * p.mean(), 2),
            "ratio": round(obs / p.sum(), 1),
            "p": round(float((np.sum(sims >= obs) + 1) / (N_SIM + 1)), 4)}


def permutation_null(d, era):
    d = d.dropna(subset=["successor_residency"])
    host, train = d.institution_id.to_numpy(), d.successor_residency.to_numpy()
    obs = int((host == train).sum())
    sims = np.array([(host == RNG.permutation(train)).sum() for _ in range(N_SIM)])
    exp = float(sims.mean())
    return {"null": "permutation", "era": era, "successions": len(d), "observed_n": obs,
            "observed_pct": round(100 * obs / len(d), 1),
            "expected_pct": round(100 * exp / len(d), 2),
            "ratio": round(obs / exp, 1) if exp else np.nan,
            "p": round(float((np.sum(sims >= obs) + 1) / (N_SIM + 1)), 4)}


ERAS = ["pre-1970", "1970-1989", "1990-2009", "2010-present"]
era4 = pd.cut(s.start, [0, 1969, 1989, 2009, 2100], labels=ERAS)
exp = {k: expectations(k) for k in ("size_all", "size_chairs", "size_chairs10")}

rows = []
for era, idx in [("all", s.index)] + [(e, s.index[era4 == e]) for e in ERAS]:
    for k, label in [("size_all", "size: all graduates"),
                     ("size_chairs", "size: graduates who became chairs"),
                     ("size_chairs10", "size: chairs appointed within 10 y")]:
        p = exp[k][idx]
        ok = ~np.isnan(p)
        rows.append(binom_null(s.loc[idx][ok], p[ok], label, era))
    rows.append(permutation_null(s.loc[idx], era))

out = pd.DataFrame(rows)
out.to_csv(TABLES / "v05_selfhire_nulls.csv", index=False)
pd.set_option("display.width", 220)
print("SELF-HIRING AGAINST FOUR NULLS\n", out.to_string(index=False))

# Ascertainment-restricted sensitivity: successions from 1970 on only.
mod = s.start >= 1970
rows = []
for era, idx in [("1970+ pooled", s.index[mod])] + [(e, s.index[(era4 == e) & mod]) for e in ERAS[1:]]:
    for k, label in [("size_all", "size: all graduates"),
                     ("size_chairs", "size: graduates who became chairs")]:
        p = exp[k][idx]
        ok = ~np.isnan(p)
        rows.append(binom_null(s.loc[idx][ok], p[ok], label, era))
    rows.append(permutation_null(s.loc[idx], era))
asc = pd.DataFrame(rows)
asc.to_csv(TABLES / "v05_selfhire_ascertainment.csv", index=False)
print("\nRESTRICTED TO SUCCESSIONS FROM 1970 (recorded graduates no longer "
      "enriched for leaders)\n", asc.to_string(index=False))
