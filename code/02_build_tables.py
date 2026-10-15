"""Build tidy tables from a raw NeurosurGen JSON snapshot.

Outputs (data/processed/):
  institutions.csv   institution_id, institution, location, country, lat, lon, n_residents_listed, is_us
  persons.csv        one row per neurosurgeon (UUID keyed)
  positions.csv      one row per chair/PD term: who, where, start, end
  training_links.csv trainee -> chair / PD at graduation, two versions:
                       site_*    : chair/PD as NeurosurGen assigns it on the person page
                       derived_* : holder of that role at the trainee's residency
                                   in the graduation year (from positions.csv)

Run:  python code/02_build_tables.py [snapshot_date]
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = sys.argv[1] if len(sys.argv) > 1 else "2026-09-13"
RAW = ROOT / "data" / "raw" / f"neurosurgen_{SNAPSHOT}"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def as_id(v):
    """Fields are sometimes an id string, sometimes an embedded object."""
    if isinstance(v, dict):
        return v.get("id")
    return v


def main():
    index = json.loads((RAW / "genealogy_index.json").read_text())
    countries = {c["id"]: c.get("name") for c in index.get("countries", [])}

    # ---- institutions + positions (institution pages) ----
    inst_rows, pos_rows = [], {}
    for f in sorted((RAW / "institution").glob("*.json")):
        res = json.loads(f.read_text())["residency"]
        inst_rows.append({
            "institution_id": res["id"],
            "institution": res["name"],
            "location": res.get("location"),
            "country": countries.get(res.get("country")),
            "lat": res.get("lat"),
            "lon": res.get("lon"),
            "n_residents_listed": len(res.get("persons", [])),
        })
        for p in res.get("positions", []):
            pos_rows[p["id"]] = {
                "position_id": p["id"],
                "person_id": as_id(p["person"]),
                "designation": p["designation"],
                "start": p.get("start"),
                "end": p.get("end"),
                "institution_id": as_id(p.get("residency")) or res["id"],
            }

    # ---- persons (person pages); also harvest any positions missing above ----
    person_rows = []
    for f in sorted((RAW / "person").glob("*.json")):
        page = json.loads(f.read_text())
        p = page.get("person")
        if not p:
            continue
        for rp in p.get("residencyPositions", []) or []:
            pos_rows.setdefault(rp["id"], {
                "position_id": rp["id"],
                "person_id": p["id"],
                "designation": rp["designation"],
                "start": rp.get("start"),
                "end": rp.get("end"),
                "institution_id": as_id(rp.get("residency")),
            })
        chair, pd_ = page.get("chair") or {}, page.get("pd") or {}
        person_rows.append({
            "person_id": p["id"],
            "name": p["name"],
            "birth": p.get("birth"),
            "death": p.get("death"),
            "graduation": p.get("graduation"),
            "gender": p.get("gender"),
            "ethnicity": p.get("ethnicity"),
            "residency_id": as_id(p.get("residency")),
            "mentor_id": as_id(p.get("mentor")),
            "site_chair_position_id": chair.get("id"),
            "site_chair_id": as_id(chair.get("person")),
            "site_pd_position_id": pd_.get("id"),
            "site_pd_id": as_id(pd_.get("person")),
            "n_mentees_listed": len(page.get("mentees", []) or []),
        })

    inst = pd.DataFrame(inst_rows)
    persons = pd.DataFrame(person_rows).drop_duplicates("person_id")
    positions = pd.DataFrame(list(pos_rows.values()))

    # merge duplicate program listings (internship pages, alternate spellings);
    # manual crosswalk of duplicate program listings: data/external/institution_aliases.csv
    aliases = pd.read_csv(ROOT / "data" / "external" / "institution_aliases.csv")
    by_name = inst.set_index("institution").institution_id
    remap = {by_name[a]: by_name[c] for a, c in zip(aliases.alias, aliases.canonical)
             if a in by_name and c in by_name}
    persons["residency_id"] = persons.residency_id.replace(remap)
    positions["institution_id"] = positions.institution_id.replace(remap)
    inst["merged_into"] = inst.institution_id.map(remap)
    inst = inst[inst.merged_into.isna()].drop(columns="merged_into")
    inst["is_us"] = inst["country"].eq("United States")
    print(f"merged {len(remap)} duplicate institution listings")
    for c in ("graduation", "birth", "death"):
        persons[c] = persons[c].astype("Int64")
    for c in ("start", "end"):
        positions[c] = positions[c].astype("Int64")
    positions["designation"] = positions["designation"].replace({"chair/chief": "chair", "director": "pd"})

    # ---- derived chair / PD at graduation from program x year ----
    links = persons[["person_id", "residency_id", "graduation",
                     "site_chair_id", "site_pd_id", "mentor_id"]].copy()
    for role in ("chair", "pd"):
        pos = positions[positions.designation == role]
        m = links.dropna(subset=["graduation"]).merge(
            pos, left_on="residency_id", right_on="institution_id", suffixes=("", "_pos"))
        # holder whose term covers the graduation year (open-ended terms run to today)
        end = m["end"].fillna(9999)
        m = m[(m["start"] <= m["graduation"]) & (m["graduation"] <= end)]
        # if two terms overlap (handover year), prefer the one that started most recently
        m = m.sort_values("start").groupby("person_id").tail(1)
        links = links.merge(
            m[["person_id", "person_id_pos"]].rename(columns={"person_id_pos": f"derived_{role}_id"}),
            on="person_id", how="left")

    inst.to_csv(OUT / "institutions.csv", index=False)
    persons.to_csv(OUT / "persons.csv", index=False)
    positions.to_csv(OUT / "positions.csv", index=False)
    links.to_csv(OUT / "training_links.csv", index=False)

    def agree(a, b):
        both = links[a].notna() & links[b].notna()
        return f"{(links.loc[both, a] == links.loc[both, b]).mean():.1%} of {both.sum()}"

    print(f"institutions {len(inst)} (US {inst.is_us.sum()})")
    print(f"persons {len(persons)}; with graduation {persons.graduation.notna().sum()}")
    print(f"positions {len(positions)}: {positions.designation.value_counts().to_dict()}; "
          f"distinct holders {positions.person_id.nunique()}")
    print("site vs derived chair agreement:", agree("site_chair_id", "derived_chair_id"))
    print("site vs derived PD agreement:   ", agree("site_pd_id", "derived_pd_id"))
    print("coverage:", links[["site_chair_id", "derived_chair_id", "site_pd_id",
                              "derived_pd_id", "mentor_id"]].notna().sum().to_dict())


if __name__ == "__main__":
    main()
