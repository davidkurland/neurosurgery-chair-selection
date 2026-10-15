"""Chairs' academic profile by era of first appointment, 1960-2026: publications before
appointment, percentile within graduating cohort, dominant subspecialty and prior NIH
funding (four-panel figure, not used in the manuscript), and the two-panel productivity
figure that is Supplemental Fig. S1. Reads v09_productivity_by_era.csv, written by
22_productivity_primary_5_10.py, so run script 22 first."""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures" / "v2"
spec = importlib.util.spec_from_file_location("f9", Path(__file__).resolve().parent / "09_figures_v2.py")
f9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f9)
BLUE, ORANGE, RAMP = f9.BLUE, f9.ORANGE, f9.RAMP
INK2, MUTED, AXIS, SURFACE, LAND = f9.INK2, f9.MUTED, f9.AXIS, f9.SURFACE, f9.LAND

A = pd.read_csv(TABLES / "v09_productivity_by_era.csv")
A["percentile_in_cohort"] = A.percentile_yrs_5_10 / 100.0
B = pd.read_csv(TABLES / "v08_chair_subspecialty_by_era.csv")
C = pd.read_csv(TABLES / "v08_chair_nih_by_era.csv")
lab = [e.replace("-", "–\n") for e in A.era]
x = np.arange(len(A))

fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))

ax = axes[0, 0]
ax.fill_between(x, A.pubs_before_q1, A.pubs_before_q3, color=LAND, lw=0)
ax.plot(x, A.pubs_before_median, color=BLUE, marker="o", ms=5, mec=SURFACE, mew=1.2)
for i in (0, len(A) - 1):
    ax.text(x[i], A.pubs_before_median[i] * 1.55, f"{A.pubs_before_median[i]:.0f}",
            ha="center", fontsize=8, color=INK2)
ax.set_yscale("log")
ax.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
ax.set_xticks(x, lab, fontsize=6.8)
ax.set_ylabel("Publications before appointment")
ax.set_title("New chairs publish far more than they used to")
ax.text(0.03, 0.93, "median, with IQR shaded", transform=ax.transAxes, fontsize=7, color=MUTED)
f9.panel_letter(ax, "A")

ax = axes[0, 1]
ax.axhline(50, color=AXIS, lw=1.2, ls=(0, (4, 2)))
ax.text(len(A) - 1, 52, "cohort median", ha="right", fontsize=7, color=MUTED)
ax.plot(x, 100 * A.percentile_in_cohort, color=ORANGE, marker="o", ms=5, mec=SURFACE, mew=1.2)
def ordinal(v):
    n = int(round(v))
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


for i in (0, len(A) - 1):
    ax.text(x[i], 100 * A.percentile_in_cohort[i] + 4,
            ordinal(100 * A.percentile_in_cohort[i]), ha="center", fontsize=8, color=INK2)
ax.set_ylim(0, 104)
ax.set_xticks(x, lab, fontsize=6.8)
ax.set_ylabel("Percentile within graduating cohort")
ax.set_title("...but sit where they always did among peers")
ax.text(0.03, 0.06, "publications in yrs 5\u201310 after graduation", transform=ax.transAxes, fontsize=7, color=MUTED)
f9.panel_letter(ax, "B")

ax = axes[1, 0]
order = ["tumor", "vascular", "spine", "functional", "trauma", "pediatric"]
names = {"tumor": "Tumor", "vascular": "Vascular", "spine": "Spine",
         "functional": "Functional", "trauma": "Trauma", "pediatric": "Pediatric"}
bottom = np.zeros(len(B))
for color, k in zip(RAMP + [ORANGE], order):
    ax.bar(x, B[k], bottom=bottom, width=0.72, color=color, lw=0, label=names[k])
    bottom += B[k].to_numpy()
ax.set_xticks(x, lab, fontsize=6.8)
ax.set_ylabel("Chairs by dominant topic (%)")
ax.set_title("What they work on has shifted")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7,
          labelcolor=INK2, handlelength=1.2)
ax.text(0.97, 0.95, "unclassified omitted", transform=ax.transAxes,
        fontsize=7, color=MUTED, ha="right")
f9.panel_letter(ax, "C")

ax = axes[1, 1]
xc = np.arange(len(C))
ax.bar(xc, C.pct_with_prior_NIH, width=0.6, color=BLUE, lw=0)
for i, v in enumerate(C.pct_with_prior_NIH):
    ax.text(xc[i], v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
ax.set_xticks(xc, [e.replace("-", "–\n") for e in C.era], fontsize=6.8)
ax.set_ylim(0, 62)
ax.set_ylabel("Chairs with NIH funding before\nappointment (%)")
ax.set_title("More arrive already NIH-funded")
ax.text(0.03, 0.93, "RePORTER covers FY1985 onward", transform=ax.transAxes,
        fontsize=7, color=MUTED)
f9.panel_letter(ax, "D")

fig.supxlabel("Era of first chair appointment", fontsize=8, color=INK2)
fig.tight_layout(h_pad=2.4, w_pad=2.0)
f9.save(fig, "figX_chair_profile")

# Two-panel version for the manuscript supplement (productivity only).
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
ax.fill_between(x, A.pubs_before_q1, A.pubs_before_q3, color=LAND, lw=0)
ax.plot(x, A.pubs_before_median, color=BLUE, marker="o", ms=5, mec=SURFACE, mew=1.2)
for i in (0, len(A) - 1):
    ax.text(x[i], A.pubs_before_median[i] * 1.6, f"{A.pubs_before_median[i]:.0f}",
            ha="center", fontsize=8, color=INK2)
ax.set_yscale("log")
ax.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
ax.set_xticks(x, lab, fontsize=6.8)
ax.set_ylabel("Publications before appointment")
ax.set_title("Output before appointment")
ax.text(0.03, 0.93, "median, with IQR shaded", transform=ax.transAxes, fontsize=7, color=MUTED)
f9.panel_letter(ax, "A")

ax = axes[1]
ax.axhline(50, color=AXIS, lw=1.2, ls=(0, (4, 2)))
ax.text(len(A) - 1, 52, "cohort median", ha="right", fontsize=7, color=MUTED)
ax.plot(x, 100 * A.percentile_in_cohort, color=ORANGE, marker="o", ms=5, mec=SURFACE, mew=1.2)
for i in (0, len(A) - 1):
    ax.text(x[i], 100 * A.percentile_in_cohort[i] + 4,
            ordinal(100 * A.percentile_in_cohort[i]), ha="center", fontsize=8, color=INK2)
ax.set_ylim(0, 104)
ax.set_xticks(x, lab, fontsize=6.8)
ax.set_ylabel("Percentile within graduating cohort")
ax.set_title("Standing among their own cohort")
ax.text(0.03, 0.06, "publications in yrs 5\u201310 after graduation", transform=ax.transAxes, fontsize=7, color=MUTED)
f9.panel_letter(ax, "B")
fig.supxlabel("Era of first chair appointment", fontsize=8, color=INK2)
fig.tight_layout(w_pad=1.8)
f9.save(fig, "figS5_chair_productivity")
