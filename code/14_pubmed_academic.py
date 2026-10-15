"""Publication-based academic-career indicator from PubMed E-utilities.

Publication activity in years 5-15 after residency graduation, with and without a
neurosurgery field restriction. Counts are queried by author name (surname and
initials) for the sample frame written by script 13 (code/13_openalex_fetch.py),
at no more than 3 requests per second.

Two esearch count calls per person, cached on disk, resumable.
No email or API key is sent.

Run:  python code/14_pubmed_academic.py
      RATE=2.5 python code/14_pubmed_academic.py
"""

import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "raw" / "pubmed_2026-09-22"
CACHE.mkdir(parents=True, exist_ok=True)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
RATE = float(os.environ.get("RATE", 2.6))
WORKERS = int(os.environ.get("WORKERS", 10))
WIN_LO, WIN_HI = 5, 15
TOOL = "nsgy_leadership_research"

NS_FILTER = ('("Neurosurgery"[Mesh] OR "Neurosurgical Procedures"[Mesh] OR neurosurg*[ad] '
             'OR neurosurg*[ti] OR neurosurg*[jour] OR "Brain Neoplasms"[Mesh] '
             'OR "Spine"[Mesh] OR "Intracranial Aneurysm"[Mesh] OR "Glioma"[Mesh] '
             'OR "Craniotomy"[Mesh] OR "Brain Injuries"[Mesh])')

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|md|phd|facs|faans|dr)\b\.?", re.I)
_lock = Lock()
_next = [0.0]


def throttle():
    with _lock:
        now = time.time()
        wait = _next[0] - now
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        _next[0] = now + 1.0 / RATE


def ascii_words(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = SUFFIX.sub(" ", s)
    return [w for w in re.sub(r"[^A-Za-z ]", " ", s).split() if w]


def pubmed_name(name):
    """'Jane Marie Doe' -> 'Doe JM'; 'Li Wei' -> 'Wei L'."""
    w = ascii_words(name)
    if len(w) < 2:
        return None
    last = w[-1]
    initials = "".join(x[0] for x in w[:-1])[:2].upper()
    return f"{last} {initials}"


def count(term, lo, hi, tag):
    key = re.sub(r"[^A-Za-z0-9]+", "_", f"{term}_{lo}_{hi}_{tag}")[:120]
    path = CACHE / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text()).get("n")
        except json.JSONDecodeError:
            pass
    q = urllib.parse.urlencode({
        "db": "pubmed", "term": term, "mindate": lo, "maxdate": hi,
        "datetype": "pdat", "rettype": "count", "retmode": "json", "tool": TOOL})
    for attempt in range(5):
        throttle()
        try:
            with urllib.request.urlopen(f"{EUTILS}?{q}", timeout=40) as r:
                d = json.loads(r.read().decode())
            n = int(d["esearchresult"]["count"])
            path.write_text(json.dumps({"n": n}))
            return n
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


def one(row):
    au = pubmed_name(row["name"])
    if not au:
        return {"person_id": row["person_id"], "matched": False}
    grad = int(row["graduation"])
    lo, hi = grad + WIN_LO, grad + WIN_HI
    any_n = count(f"{au}[au]", lo, hi, "any")
    ns_n = count(f"{au}[au] AND {NS_FILTER}", lo, hi, "ns")
    return {"person_id": row["person_id"], "name": row["name"], "pubmed_name": au,
            "graduation": grad, "win_lo": lo, "win_hi": hi,
            "pubs_any": any_n, "pubs_ns": ns_n,
            "matched": any_n is not None}


def main():
    frame = PROC / "openalex_sample_frame.csv"
    if not frame.exists():
        sys.exit("run code/13_openalex_fetch.py first to write the sample frame")
    g = pd.read_csv(frame)
    print(f"sample: {len(g)} graduates "
          f"({(g.gender == 'female').sum()} women)", flush=True)
    rows = g.to_dict("records")
    out = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, rec in enumerate(ex.map(one, rows), 1):
            out.append(rec)
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
                pd.DataFrame(out).to_csv(PROC / "pubmed_academic.csv", index=False)
    df = pd.DataFrame(out)
    df = df.merge(g[["person_id", "gender", "residency_id", "era"]], on="person_id", how="left")
    df.to_csv(PROC / "pubmed_academic.csv", index=False)
    print(f"\nwrote pubmed_academic.csv ({len(df)} rows)")
    ok = df[df.matched]
    for thr in (1, 3, 5):
        print(f"  academic (>={thr} neurosurgery pubs in window): "
              f"{100 * (ok.pubs_ns >= thr).mean():.1f}%  "
              f"women {100 * (ok[ok.gender == 'female'].pubs_ns >= thr).mean():.1f}%  "
              f"men {100 * (ok[ok.gender != 'female'].pubs_ns >= thr).mean():.1f}%")


if __name__ == "__main__":
    main()
