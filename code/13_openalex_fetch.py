"""Fetch OpenAlex author records and windowed publication counts for NeurosurGen graduates.

Builds a publication-based "academic career" indicator, an alternative to the
all-graduates denominator in the self-hiring and women-vs-pipeline comparisons.
The stratified sample frame is written to data/processed/openalex_sample_frame.csv
and is also the input of script 14.

Two calls per person, both cached on disk so the run is resumable:
  1. /authors?display_name.search  -> candidate authors, scored and disambiguated here
  2. /works?author.id,publication_year=grad+5..grad+15&group_by=institutions.type

Requests are anonymous (no mailto parameter).

Run:  python code/13_openalex_fetch.py            # graduates 1970-2015
      CONCURRENCY=8 python code/13_openalex_fetch.py
"""

import asyncio
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from curl_cffi.requests import AsyncSession

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "raw" / "openalex_2026-09-22"
(CACHE / "authors").mkdir(parents=True, exist_ok=True)
(CACHE / "works").mkdir(parents=True, exist_ok=True)

API = "https://api.openalex.org"
CONCURRENCY = int(os.environ.get("CONCURRENCY", 2))
RATE = float(os.environ.get("RATE", 1.1))          # requests per second, global
SAMPLE_MEN = int(os.environ.get("SAMPLE_MEN", 1600))
GRAD_LO, GRAD_HI = 1970, 2015
WIN_LO, WIN_HI = 5, 15  # years after graduation

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|md|phd|facs|faans|dr)\b\.?", re.I)
NEURO = ("neurosurg", "neurolog", "brain", "spine", "spinal", "glioma", "aneurysm",
         "meningioma", "skull", "epilepsy", "cranial", "tumor", "trauma", "stroke",
         "surgery", "surgical", "neuro")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = SUFFIX.sub(" ", s)
    return re.sub(r"[^a-z ]", " ", s.lower()).split()


def name_score(query, cand):
    """-1 = incompatible; higher is better."""
    q, c = norm(query), norm(cand)
    if not q or not c or q[-1] != c[-1]:
        return -1
    qf, cf = q[0], c[0]
    if qf == cf:
        s = 3
    elif len(qf) == 1 or len(cf) == 1:
        s = 2 if qf[0] == cf[0] else -1
    else:
        return -1
    qm = {x[0] for x in q[1:-1]}
    cm = {x[0] for x in c[1:-1]}
    if qm and cm:
        s += 1 if qm & cm else -1
    return s


def middle_conflict(query, cand):
    q, c = norm(query), norm(cand)
    qm = {x[0] for x in q[1:-1]}
    cm = {x[0] for x in c[1:-1]}
    return bool(qm and cm and not (qm & cm))


def score_candidate(person_name, grad, a):
    ns = name_score(person_name, a.get("display_name", ""))
    if ns < 0:
        return None
    s = float(ns)
    topics = " ".join(t.get("display_name", "") for t in (a.get("topics") or [])).lower()
    fields = " ".join((t.get("field") or {}).get("display_name", "") for t in (a.get("topics") or [])).lower()
    if any(k in topics for k in NEURO):
        s += 3
    if "medicine" in fields or "health" in fields or "neuroscience" in fields:
        s += 1
    yrs = [y for x in (a.get("affiliations") or []) for y in (x.get("years") or [])]
    if yrs and any(grad - 3 <= y <= grad + 45 for y in yrs):
        s += 1
    if yrs and max(yrs) < grad - 3:
        s -= 3
    cc = {(x.get("institution") or {}).get("country_code") for x in (a.get("affiliations") or [])}
    if "US" in cc or "CA" in cc:
        s += 1
    if (a.get("works_count") or 0) == 0:
        s -= 2
    return s


class Limiter:
    """Global rate limiter (OpenAlex rate-limits anonymous requests)."""

    def __init__(self, rate):
        self.gap = 1.0 / rate
        self.next = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            if now < self.next:
                await asyncio.sleep(self.next - now)
                now = asyncio.get_event_loop().time()
            self.next = now + self.gap


LIMIT = Limiter(RATE)


async def get(sess, url, path, tries=6):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    for i in range(tries):
        try:
            await LIMIT.wait()
            r = await sess.get(url, timeout=45)
            if r.status_code == 200:
                d = r.json()
                path.write_text(json.dumps(d))
                return d
            if r.status_code in (429, 503):
                await asyncio.sleep(min(60, 5 * 2 ** i))
                continue
            path.write_text(json.dumps({"_error": r.status_code}))
            return {"_error": r.status_code}
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                return {"_error": str(e)}
            await asyncio.sleep(2 * (i + 1))
    return {"_error": "retries"}


SELECT = "id,display_name,works_count,cited_by_count,affiliations,topics"


def slug(q):
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:80] or "_"


def query_name(name):
    """First + last token only. OpenAlex search requires every token to appear, and
    NeurosurGen often stores a full middle name where OpenAlex stores an initial."""
    w = [x for x in SUFFIX.sub(" ", str(name)).replace(".", " ").split() if x]
    if len(w) < 2:
        return " ".join(w)
    return f"{w[0]} {w[-1]}"


async def one(sess, sem, row, out):
    pid, name, grad = row.person_id, row.name, int(row.graduation)
    async with sem:
        q = query_name(name).replace("&", " ")
        url = f"{API}/authors?filter=display_name.search:{q},affiliations.institution.country_code:us|ca&per-page=25&select={SELECT}"
        d = await get(sess, url, CACHE / "authors" / f"{slug(q)}.json")
        results = d.get("results") or []
        if not results:
            url2 = f"{API}/authors?filter=display_name.search:{q}&per-page=25&select={SELECT}"
            d = await get(sess, url2, CACHE / "authors" / f"{slug(q)}_nc.json")
            results = d.get("results") or []

        scored = []
        for a in results:
            s = score_candidate(name, grad, a)
            if s is not None:
                scored.append((s, a))
        scored.sort(key=lambda x: (-x[0], -(x[1].get("works_count") or 0)))

        rec = {"person_id": pid, "name": name, "query": q, "graduation": grad,
               "n_candidates": len(results), "n_name_ok": len(scored)}
        if not scored:
            rec["matched"] = False
            out.append(rec)
            return
        best_s, best = scored[0]
        rec |= {"matched": True, "author_id": best["id"].rsplit("/", 1)[-1],
                "author_name": best.get("display_name"), "score": best_s,
                "works_total": best.get("works_count"),
                "ambiguous": len(scored) > 1 and scored[1][0] >= best_s - 1,
                "middle_conflict": middle_conflict(name, best.get("display_name", ""))}

        lo, hi = grad + WIN_LO, grad + WIN_HI
        aid = rec["author_id"]
        wurl = f"{API}/works?filter=author.id:{aid},publication_year:{lo}-{hi}&group_by=institutions.type"
        w = await get(sess, wurl, CACHE / "works" / f"{aid}_{lo}_{hi}.json")
        groups = {g["key"]: g["count"] for g in (w.get("group_by") or [])}
        rec |= {"win_lo": lo, "win_hi": hi,
                "works_window": int((w.get("meta") or {}).get("count") or 0),
                "w_education": groups.get("education", 0),
                "w_healthcare": groups.get("healthcare", 0),
                "w_company": groups.get("company", 0),
                "w_other": sum(v for k, v in groups.items()
                               if k not in ("education", "healthcare", "company"))}
        out.append(rec)


async def main():
    per = pd.read_csv(PROC / "persons.csv")
    inst = pd.read_csv(PROC / "institutions.csv")
    NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
    g = per[per.residency_id.isin(NA) & per.graduation.notna()]
    g = g[(g.graduation >= GRAD_LO) & (g.graduation <= GRAD_HI)].copy()
    full = len(g)

    # Stratified sample: every woman, plus men sampled within graduation era.
    # The denominator only needs P(academic | sex, era); sampling men is sufficient
    # and the era counts themselves come from the complete NeurosurGen cohort.
    g["era"] = pd.cut(g.graduation, [0, 1979, 1989, 1999, 2009, 2100],
                      labels=["1970s", "1980s", "1990s", "2000s", "2010-15"])
    women = g[g.gender == "female"]
    men = g[g.gender != "female"]
    frac = min(1.0, SAMPLE_MEN / max(1, len(men)))
    men_s = men.groupby("era", observed=True, group_keys=False).apply(
        lambda d: d.sample(max(1, round(len(d) * frac)), random_state=13))
    g = pd.concat([women, men_s]).sort_values("graduation")
    g.to_csv(PROC / "openalex_sample_frame.csv", index=False)
    print(f"cohort {full} graduates {GRAD_LO}-{GRAD_HI}; sampled {len(g)} "
          f"({len(women)} women, {len(men_s)} of {len(men)} men)", flush=True)

    out = []
    sem = asyncio.Semaphore(CONCURRENCY)
    async with AsyncSession(impersonate="chrome") as sess:
        tasks = [one(sess, sem, r, out) for r in g.itertuples()]
        for i in range(0, len(tasks), 100):
            await asyncio.gather(*tasks[i:i + 100])
            print(f"  {min(i + 100, len(tasks))}/{len(tasks)} done", flush=True)
            pd.DataFrame(out).to_csv(PROC / "openalex_authors.csv", index=False)

    df = pd.DataFrame(out)
    df["sampled_sex"] = df.person_id.map(dict(zip(g.person_id, g.gender)))
    dest = PROC / "openalex_authors.csv"
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest} ({len(df)} rows)")
    print("matched:", int(df.matched.sum()), f"({100 * df.matched.mean():.1f}%)")
    if "ambiguous" in df:
        print("ambiguous among matched:", int(df.ambiguous.fillna(False).sum()))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
