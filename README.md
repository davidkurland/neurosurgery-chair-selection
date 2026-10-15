# Neurosurgery chair selection over a century

Analysis code and derived tables.

This repository accompanies the manuscript *From Apprentice to Executive: A Century of Neurosurgical Chair Selection* (Kurland, Gajjar, Asaad, Alleyne,
Kondziolka, Golfinos, Oermann, Gokaslan). Each tagged release is archived on Zenodo with its own DOI.

## Contents

- `code/`: the analysis pipeline, scripts 01 to 24, 26 and 27, with `code/environment.yml`. See `code/README.md`
  for what each script does and the run order.
- `data/tables/`: 59 aggregate output tables (by era, program, model or year). Tables 1 to 3
  and every figure in the manuscript are built from these files or from the supplemental tables.
- `data/external/institution_aliases.csv`: the institution-name corrections applied when building
  the tables; `data/external/ne_50m_*.geojson`: Natural Earth 1:50m country and state boundaries (public
  domain, naturalearthdata.com) used by the map in Figure 1.
- `supplement/`: the supplemental tables as CSV and the supplemental figures (PNG and PDF), under
  their manuscript numbers:
- `supplement/Supplemental_Table_S1_career_stage_vs_workforce.csv`
- `supplement/Supplemental_Table_S2_publications_before_first_chair.csv`
- `supplement/Supplemental_Table_S3_tenure_hazard_age_adjusted.csv`
- `supplement/Supplemental_Table_S4_selfhire_vs_expectations.csv`
- `supplement/Supplemental_Table_S5_selfhire_vs_prolific_programs.csv`
- `supplement/Supplemental_Table_S6_chairs_per_graduate_by_era.csv`
- `supplement/Supplemental_Table_S7_program_placement_ranking.csv`
- `supplement/Supplemental_Table_S8_women_chair_pipeline.csv`
- `supplement/Supplemental_Table_S9_women_pd_pipeline.csv`

## Data source and access

The data come from NeurosurGen.com (September 2026 snapshot), collected with the permission of
NeurosurGen Inc., which maintains the database. Person-level records are not redistributed here: the raw
page snapshots, the person, position and training-link tables built from them, the publication counts
per person, and output tables that list individuals by name or id. The analytic dataset is available
from the corresponding author on reasonable request.

## Reproduction

Create the environment from `code/environment.yml`. With the person-level tables in `data/processed/`
(on request, or rebuilt with scripts 01 and 02), run the scripts from the repository root in numerical
order, except that script 22 runs before script 20. Scripts 13, 14, 17, 19 and 21 query OpenAlex, PubMed
and NIH RePORTER and cache their results. The scripts write to `outputs/tables` and `outputs/figures`;
the aggregate tables the manuscript uses are the copies in `data/tables`. Table names prefixed `v03` to
`v10` are numbered by the analysis stage that produced them. Supplemental Table S7 is the placement
ranking in `data/tables/program_ranking_chair.csv` formatted for the manuscript. Tables that name
individuals are limited to the chairs of record already named in the manuscript (Table 3).

## Citing

Cite the archived release, not the repository URL. Each release is archived on Zenodo, and its DOI will be
added to `CITATION.cff` and this file when the first release is created.

## Licence

Code in `code/` is MIT licensed (`LICENSE`). Tables and figures in `data/` and `supplement/` are
CC BY 4.0 (`LICENSE-data`).

Corresponding author: David B. Kurland MD, PhD, Department of Neurosurgery, Brown University Health,
Providence, RI (dkurland@brownhealth.org).
