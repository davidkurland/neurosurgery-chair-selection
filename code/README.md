# Code

Python 3.12. Run from the repository root in numerical order, except that script 22 runs before
script 20. Scripts are numbered in the order the analysis was built; later scripts refine or extend earlier
ones. Outputs go to `outputs/`; the tables the manuscript uses are copied to `data/tables/`. Script 09
draws the manuscript's figures and the supplemental figures; script 05 draws an earlier set that the
manuscript does not use, and script 20 draws Supplemental Fig. S1 and a companion figure. Script 07 writes
sensitivity analyses of the placement order reported in the Supplemental Results.

| Script | Purpose |
|---|---|
| `01_scrape_neurosurgen.py` | Scrape NeurosurGen.com (with permission) into raw JSON snapshots. |
| `02_build_tables.py` | Build tidy tables from a raw NeurosurGen JSON snapshot. |
| `03_placement_hierarchy.py` | Placement hierarchy of neurosurgery residency programs. |
| `04_leadership_inheritance.py` | Is academic leadership inherited through training lineage? |
| `05_figures.py` | Manuscript figures (PNG 300 dpi + PDF) from outputs/tables. |
| `06_robustness.py` | Robustness analyses, run after scripts 03 and 04. |
| `07_followup_program_era.py` | Sensitivity analyses for the within-program leader-maker association: program-era |
| `08_chair_lifecycle.py` | The changing path to the chair: tenure, age at appointment, succession and chairmaker analyses. |
| `09_figures_v2.py` | Main and supplementary figures (PNG 300 dpi + PDF) -> outputs/figures/v2/ |
| `10_v03_checks.py` | Robustness checks: mentor concentration, women versus pipeline lag sensitivity, early-era ascertainment. |
| `11_v04_selfhire_successions.py` | Self-hiring measured among successions (founding chairs excluded). |
| `12_v05_tougher_nulls.py` | Self-hiring against four nulls, plus a sensitivity analysis restricted to successions |
| `13_openalex_fetch.py` | Fetch OpenAlex author records and windowed publication counts for NeurosurGen graduates. |
| `14_pubmed_academic.py` | Publication-based academic-career indicator from PubMed E-utilities. |
| `15_v05_women_denominators.py` | Women's chair appointments against three pipeline denominators. |
| `16_v06_strengthen.py` | Career stage, age-adjusted tenure, self-hiring concentration, classifier validation, |
| `17_chair_profile_fetch.py` | Fetch the academic profile of every chair: productivity, subspecialty, NIH funding. |
| `18_chair_profile_analysis.py` | What kind of neurosurgeon becomes a chair: productivity, subspecialty, funding. |
| `19_pubmed_reference_pre1970.py` | Extend the non-chair publication reference to graduates from 1945-1969. |
| `20_figure_chair_profile.py` | Chairs' academic profile by era of first appointment, 1960-2026: publications before |
| `21_pubmed_window_5_10.py` | Publications in years 5-10 after graduation, for chairs and both reference cohorts. |
| `22_productivity_primary_5_10.py` | Productivity percentile on the years 5-10 window, with 5-15 as sensitivity. |
| `23_v09_selfhire_asc_chairs10.py` | Ascertainment-restricted self-hiring: third size null (chairs appointed within 10 y). |
| `24_v09_audit_gaps.py` | Supplementary numbers cited in the manuscript: KM median tenure, self-hire tests, |
| `26_v10_prolific_programs_checks.py` | Two robustness checks on program size and chair production. |
| `27_v10_traced_numbers.py` | Save numbers cited in the manuscript that other scripts do not tabulate. |
