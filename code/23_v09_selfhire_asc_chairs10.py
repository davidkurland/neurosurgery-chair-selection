"""Ascertainment-restricted self-hiring: third size null (chairs appointed within 10 y).

code/12_v05_tougher_nulls.py writes `v05_selfhire_ascertainment.csv` with two size
nulls plus the permutation null. This script adds the `size_chairs10` null (program's
share of chairs first appointed within +/-10 y of the succession) on the same restricted
successions, using the construction of code/12_v05_tougher_nulls.py with its own
random generator (seed 23).

Run:  python code/23_v09_selfhire_asc_chairs10.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
RNG = np.random.default_rng(23)
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

exp = []
for r in s.itertuples():
    w = (gy >= r.start - 25) & (gy <= r.start - 5)
    m = w & ec & (np.abs(np.nan_to_num(fc, nan=9e9) - r.start) <= 10)
    exp.append((prog[m] == r.institution_id).mean() if m.sum() else np.nan)
exp = np.array(exp)

ERAS = ["1970-1989", "1990-2009", "2010-present"]
era4 = pd.cut(s.start, [0, 1969, 1989, 2009, 2100],
              labels=["pre-1970"] + ERAS)
mod = s.start >= 1970

rows = []
for era, idx in [("1970+ pooled", s.index[mod])] + \
                [(e, s.index[(era4 == e) & mod]) for e in ERAS]:
    p = exp[idx]
    ok = ~np.isnan(p)
    d, p = s.loc[idx][ok], p[ok]
    obs = int(d.own_graduate.sum())
    sims = RNG.binomial(1, np.tile(p, (N_SIM, 1))).sum(1)
    rows.append({"null": "size: chairs appointed within 10 y", "era": era,
                 "successions": len(d), "observed_n": obs,
                 "observed_pct": round(100 * obs / len(d), 1),
                 "expected_pct": round(100 * p.mean(), 2),
                 "ratio": round(obs / p.sum(), 1),
                 "p": round(float((np.sum(sims >= obs) + 1) / (N_SIM + 1)), 4)})

out = pd.DataFrame(rows)
out.to_csv(TABLES / "v09_selfhire_asc_chairs10.csv", index=False)
pd.set_option("display.width", 220)
print("ASCERTAINMENT-RESTRICTED, CHAIRS +/-10 Y NULL\n", out.to_string(index=False))
