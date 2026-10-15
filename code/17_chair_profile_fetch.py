"""Fetch the academic profile of every chair: productivity, subspecialty, NIH funding.

A  PubMed publication counts per chair: career-to-appointment, and the years 5-15
   after graduation window used for the non-chair comparison sample (14_pubmed_academic.py).
B  Subspecialty from MeSH: counts in 6 topic areas over the chair's career.
C  NIH RePORTER: projects with the chair as principal investigator, by fiscal year,
   so awards before appointment can be counted. RePORTER covers FY1985 onward.

All three are cached on disk and resumable. No email or API key is sent.

Run:  python code/17_chair_profile_fetch.py
      PART=C python code/17_chair_profile_fetch.py
"""

import hashlib
import json
import os
import re
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
CACHE = ROOT / "data" / "raw" / "chair_profile_2026-09-23"
for sub in ("pubs", "mesh", "nih"):
    (CACHE / sub).mkdir(parents=True, exist_ok=True)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
REPORTER = "https://api.reporter.nih.gov/v2/projects/search"
TOOL = "nsgy_leadership_research"
PART = os.environ.get("PART", "ABC")

NS = ('("Neurosurgery"[Mesh] OR "Neurosurgical Procedures"[Mesh] OR neurosurg*[ad] '
      'OR neurosurg*[ti] OR neurosurg*[jour] OR "Brain Neoplasms"[Mesh] OR "Spine"[Mesh] '
      'OR "Intracranial Aneurysm"[Mesh] OR "Glioma"[Mesh] OR "Craniotomy"[Mesh] '
      'OR "Brain Injuries"[Mesh])')

SUBSPECIALTY = {
    "tumor": '("Brain Neoplasms"[Mesh] OR "Glioma"[Mesh] OR "Meningioma"[Mesh] OR "Skull Base Neoplasms"[Mesh] OR "Pituitary Neoplasms"[Mesh])',
    "vascular": '("Intracranial Aneurysm"[Mesh] OR "Intracranial Arteriovenous Malformations"[Mesh] OR "Cerebral Revascularization"[Mesh] OR "Subarachnoid Hemorrhage"[Mesh])',
    "spine": '("Spinal Diseases"[Mesh] OR "Spinal Fusion"[Mesh] OR "Intervertebral Disc Displacement"[Mesh] OR "Spinal Cord Injuries"[Mesh])',
    "functional": '("Deep Brain Stimulation"[Mesh] OR "Epilepsy"[Mesh] OR "Stereotaxic Techniques"[Mesh] OR "Radiosurgery"[Mesh] OR "Pain Management"[Mesh])',
    "pediatric": '("Hydrocephalus"[Mesh] OR "Neural Tube Defects"[Mesh] OR "Craniosynostoses"[Mesh] OR "Pediatrics"[Mesh])',
    "trauma": '("Craniocerebral Trauma"[Mesh] OR "Brain Injuries"[Mesh] OR "Intracranial Pressure"[Mesh] OR "Critical Care"[Mesh])',
}

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|md|phd|facs|faans|dr)\b\.?", re.I)
_lock, _next = Lock(), [0.0]
RATE = float(os.environ.get("RATE", 2.6))


def throttle(rate=None):
    gap = 1.0 / (rate or RATE)
    with _lock:
        now = time.time()
        if _next[0] > now:
            time.sleep(_next[0] - now)
            now = time.time()
        _next[0] = now + gap


def words(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return [w for w in re.sub(r"[^A-Za-z ]", " ", SUFFIX.sub(" ", s)).split() if w]


def pubmed_name(name):
    w = words(name)
    if len(w) < 2:
        return None
    return f"{w[-1]} {''.join(x[0] for x in w[:-1])[:2].upper()}"


def count(term, lo, hi, sub, tag):
    # Hash the term: the MeSH filters are long, so a truncated key would collide
    # across windows and topics for the same author.
    h = hashlib.md5(term.encode()).hexdigest()[:12]
    path = CACHE / sub / f"{tag}_{lo}_{hi}_{h}.json"
    if path.exists():
        try:
            return json.loads(path.read_text()).get("n")
        except json.JSONDecodeError:
            pass
    q = urllib.parse.urlencode({"db": "pubmed", "term": term, "mindate": lo, "maxdate": hi,
                                "datetype": "pdat", "rettype": "count", "retmode": "json",
                                "tool": TOOL})
    for a in range(5):
        throttle()
        try:
            with urllib.request.urlopen(f"{EUTILS}?{q}", timeout=40) as r:
                n = int(json.loads(r.read().decode())["esearchresult"]["count"])
            path.write_text(json.dumps({"n": n}))
            return n
        except Exception:  # noqa: BLE001
            time.sleep(2 * (a + 1))
    return None


def nih(name, person_id):
    path = CACHE / "nih" / f"{person_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    w = words(name)
    if len(w) < 2:
        return {"_skip": True}
    payload = json.dumps({
        "criteria": {"pi_names": [{"any_name": f"{w[0]} {w[-1]}"}]},
        "include_fields": ["FiscalYear", "PrincipalInvestigators", "Organization",
                           "AwardAmount", "ProjectNum"],
        "limit": 500, "offset": 0}).encode()
    for a in range(4):
        throttle(1.2)
        try:
            req = urllib.request.Request(REPORTER, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode())
            last, init = w[-1].lower(), w[0][0].lower()
            rows = []
            for p in d.get("results", []):
                for pi in (p.get("principal_investigators") or []):
                    nm = words(pi.get("full_name", ""))
                    if nm and nm[-1].lower() == last and nm[0][:1].lower() == init:
                        rows.append({"fy": p.get("fiscal_year"),
                                     "amount": p.get("award_amount"),
                                     "num": p.get("project_num"),
                                     "org": (p.get("organization") or {}).get("org_name"),
                                     "exact_first": nm[0].lower() == w[0].lower()})
                        break
            out = {"n_total_hits": d.get("meta", {}).get("total"), "projects": rows}
            path.write_text(json.dumps(out))
            return out
        except Exception:  # noqa: BLE001
            time.sleep(3 * (a + 1))
    return {"_error": True}


def main():
    per = pd.read_csv(PROC / "persons.csv")
    inst = pd.read_csv(PROC / "institutions.csv")
    NA = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
    pos = pd.read_csv(PROC / "positions.csv")
    ch = (pos[(pos.designation == "chair") & pos.institution_id.isin(NA)]
          .dropna(subset=["start"]).sort_values("start").drop_duplicates("person_id"))
    ch = ch.merge(per[["person_id", "name", "graduation", "gender", "birth"]],
                  on="person_id", how="left")
    ch = ch[ch.graduation.notna()]
    ch = ch[ch.start >= 1960]           # MEDLINE coverage is thin before this
    ch["au"] = ch.name.map(pubmed_name)
    print(f"chairs with a first appointment from 1960: {len(ch)}", flush=True)

    def job(r):
        g, y, au = int(r["graduation"]), int(r["start"]), r["au"]
        out = {"person_id": r["person_id"], "name": r["name"], "graduation": g,
               "appointment": y, "gender": r["gender"], "pubmed_name": au}
        if not au:
            return out
        if "A" in PART:
            out["pubs_before"] = count(f"{au}[au] AND {NS}", g, max(g, y - 1), "pubs", "pre")
            out["pubs_5_15"] = count(f"{au}[au] AND {NS}", g + 5, g + 15, "pubs", "w515")
            out["pubs_career"] = count(f"{au}[au] AND {NS}", g, g + 40, "pubs", "car")
        if "B" in PART:
            for k, f in SUBSPECIALTY.items():
                out[f"ss_{k}"] = count(f"{au}[au] AND {f}", g, g + 40, "mesh", k)
        if "C" in PART and y >= 1988:
            d = nih(r["name"], r["person_id"])
            pr = d.get("projects", [])
            before = [p for p in pr if p.get("fy") and p["fy"] < y]
            out |= {"nih_any": len(pr) > 0, "nih_before_n": len(before),
                    "nih_before_any": len(before) > 0,
                    "nih_first_fy": min([p["fy"] for p in pr], default=None),
                    "nih_exact_first_n": sum(1 for p in before if p.get("exact_first"))}
        return out

    rows = ch.to_dict("records")
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, rec in enumerate(ex.map(job, rows), 1):
            out.append(rec)
            if i % 100 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
                pd.DataFrame(out).to_csv(PROC / "chair_profile.csv", index=False)
    df = pd.DataFrame(out)
    df.to_csv(PROC / "chair_profile.csv", index=False)
    print(f"\nwrote chair_profile.csv ({len(df)} rows)")
    if "pubs_before" in df:
        print(df.groupby(pd.cut(df.appointment, [0, 1989, 2009, 2100],
                                labels=["1960-89", "1990-2009", "2010-26"]),
                         observed=True).pubs_before.median())


if __name__ == "__main__":
    main()
