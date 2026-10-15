"""Self-hiring measured among successions (founding chairs excluded).

For each succession, the expectation for "new chair graduated from this program"
is the program's share of all North American graduates 5-25 years before the
appointment. Observed counts vs sum of expectations, simulated (5,000 draws).
The expectation is also written back to lifecycle_successions.csv as the column
p_own_size, so that file gains this column once this script has run.

Run:  python code/11_v04_selfhire_successions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
RNG = np.random.default_rng(4)

s = pd.read_csv(TABLES / "lifecycle_successions.csv")
inst = pd.read_csv(PROC / "institutions.csv")
NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
per = pd.read_csv(PROC / "persons.csv")
g = per[per.residency_id.isin(NA) & per.graduation.notna()]
gy, prog = g.graduation.to_numpy(), g.residency_id.to_numpy()

p = []
for r in s.itertuples():
    m = (gy >= r.start - 25) & (gy <= r.start - 5)
    p.append((prog[m] == r.institution_id).mean() if m.sum() else np.nan)
s["p_own_size"] = p
s.to_csv(TABLES / "lifecycle_successions.csv", index=False)


def summarize(groups, label):
    rows = []
    for era, d in groups:
        d = d.dropna(subset=["p_own_size"])
        obs = int(d.own_graduate.sum())
        sims = RNG.binomial(1, np.tile(d.p_own_size.to_numpy(), (5000, 1))).sum(1)
        rows.append({"era": era, "successions": len(d), "own_graduate_n": obs,
                     "own_graduate_pct": round(100 * obs / len(d), 1),
                     "expected_pct": round(100 * d.p_own_size.mean(), 2),
                     "ratio": round(obs / d.p_own_size.sum(), 1),
                     "p": float((np.sum(sims >= obs) + 1) / 5001)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"v04_selfhire_successions_{label}.csv", index=False)
    return out


era5 = pd.cut(s.start, [0, 1949, 1969, 1989, 2009, 2100], labels=["<1950", "1950-1969", "1970-1989", "1990-2009", "2010-2026"])
era4 = pd.cut(s.start, [0, 1969, 1989, 2009, 2100], labels=["pre-1970", "1970-1989", "1990-2009", "2010-present"])
pd.set_option("display.width", 200)
print(summarize(s.groupby(era5, observed=True), "5era").to_string(index=False))
print(summarize(s.groupby(era4, observed=True), "4era").to_string(index=False))
print(summarize([("all", s), ("1970+", s[s.start >= 1970])], "pooled").to_string(index=False))
