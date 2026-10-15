"""Scrape NeurosurGen.com (with permission) into raw JSON snapshots.

Every page on the site is a Next.js page whose server-rendered data lives in
<script id="__NEXT_DATA__">. We save `props.pageProps` verbatim, one file per
page, so the raw snapshot is never edited and parsing happens downstream.

Crawl order:
  1. /genealogy                -> index of institutions
  2. /institution/<id>         -> residents (persons) + chair/PD positions
  3. /person/<id>              -> chair, PD, mentees, positions for each person
  4. any person id referenced on a person page but not yet fetched

Resumable: pages already on disk are skipped.

Run:  python code/01_scrape_neurosurgen.py [snapshot_date]
"""

import asyncio
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from curl_cffi.requests import AsyncSession

BASE_URL = "https://www.neurosurgen.com"
CONCURRENCY = int(os.environ.get("SCRAPE_CONCURRENCY", 16))
DELAY_S = float(os.environ.get("SCRAPE_DELAY_S", 0.05))
RETRIES = 4

SNAPSHOT = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / f"neurosurgen_{SNAPSHOT}"
NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


async def fetch_page_props(session, sem, path):
    for attempt in range(RETRIES):
        async with sem:
            try:
                r = await session.get(BASE_URL + path, impersonate="chrome", timeout=60)
                await asyncio.sleep(DELAY_S)
                if r.status_code == 200:
                    m = NEXT_DATA.search(r.text)
                    if m:
                        return json.loads(m.group(1))["props"]["pageProps"]
                elif r.status_code == 404:
                    return None
            except Exception as e:  # network hiccup; back off and retry
                print(f"  {path}: {e!r}")
        await asyncio.sleep(2 ** attempt * 2)
    print(f"  FAILED {path}")
    return None


async def fetch_to_disk(session, sem, kind, id_, failures):
    f = OUT / kind / f"{id_}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:  # truncated by an interrupted run; refetch
            pass
    props = await fetch_page_props(session, sem, f"/{kind}/{id_}")
    if props is None:
        failures.append(f"{kind}/{id_}")
        return None
    f.write_text(json.dumps(props))
    return props


def referenced_person_ids(obj):
    """All UUIDs stored under keys that point at people, anywhere in a payload."""
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("person", "mentor", "chair", "pd") and isinstance(v, str) and UUID.match(v):
                found.add(v)
            if k in ("person", "chair", "pd") and isinstance(v, dict) and UUID.match(str(v.get("id", ""))):
                found.add(v["id"])
            found |= referenced_person_ids(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= referenced_person_ids(v)
    return found


async def gather_progress(coros, label):
    results, done = [], 0
    for chunk_start in range(0, len(coros), 200):
        results += await asyncio.gather(*coros[chunk_start:chunk_start + 200])
        done += len(coros[chunk_start:chunk_start + 200])
        print(f"{label}: {done}/{len(coros)}", flush=True)
    return results


async def main():
    (OUT / "institution").mkdir(parents=True, exist_ok=True)
    (OUT / "person").mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    failures = []

    async with AsyncSession() as session:
        index_file = OUT / "genealogy_index.json"
        if index_file.exists():
            index = json.loads(index_file.read_text())
        else:
            index = await fetch_page_props(session, sem, "/genealogy")
            index_file.write_text(json.dumps(index))
        inst_ids = [i["id"] for i in index["institutions"]]
        print(f"{len(inst_ids)} institutions", flush=True)

        insts = await gather_progress(
            [fetch_to_disk(session, sem, "institution", i, failures) for i in inst_ids], "institutions")

        person_ids = set()
        for inst in filter(None, insts):
            res = inst.get("residency", {})
            person_ids |= {p["id"] for p in res.get("persons", [])}
            person_ids |= referenced_person_ids(res.get("positions", []))
        print(f"{len(person_ids)} persons from institution pages", flush=True)

        fetched = set()
        while person_ids - fetched:
            batch = sorted(person_ids - fetched)
            pages = await gather_progress(
                [fetch_to_disk(session, sem, "person", p, failures) for p in batch], "persons")
            fetched |= set(batch)
            for page in filter(None, pages):
                person_ids |= referenced_person_ids(page)

    meta = {
        "snapshot": SNAPSHOT,
        "n_institutions": len(inst_ids),
        "n_persons_fetched": len(list((OUT / "person").glob("*.json"))),
        "failures": failures,
        "site_stats": index.get("stats"),
    }
    (OUT / "scrape_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1)[:2000])


if __name__ == "__main__":
    asyncio.run(main())
