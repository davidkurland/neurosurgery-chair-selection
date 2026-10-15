"""Manuscript figures (PNG 300 dpi + PDF) from outputs/tables.

Fig 1  Concentration of chair production (Lorenz curve)
Fig 2  Placement hierarchy: origin vs destination program rank percentile
Fig 3  Top-ranked programs with bootstrap rank intervals
Fig 4  Era trends: self-hire rate, top-10 program share, share of chairs who are women
Fig 5  Cumulative incidence of chair appointment by leader-maker exposure
Fig 6  Forest plot: leader-maker hazard ratio across models

Figures 2-4 and 6 are drawn from tables in outputs/tables; Figures 1 and 5 also use the
person-level tables in data/processed.
Run:  python code/05_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.duration.survfunc import SurvfuncRight

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Figure palette (light theme): three series colours, ink/text greys, grid, axis and surface colours
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "lines.linewidth": 2, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png", dpi=300)
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)


def fig1_lorenz():
    edges = pd.read_csv(TABLES / "placement_edges_chair.csv")
    persons = pd.read_csv(ROOT / "data" / "processed" / "persons.csv")
    inst = pd.read_csv(ROOT / "data" / "processed" / "institutions.csv")
    na = set(inst.loc[inst.country.isin(["United States", "Canada"]), "institution_id"])
    progs = persons[persons.residency_id.isin(na) & persons.graduation.notna()].residency_id.unique()
    prod = pd.Series(0, index=progs).add(edges.groupby("source").size(), fill_value=0)
    prod = prod.sort_values(ascending=False).to_numpy()
    x = np.arange(0, len(prod) + 1) / len(prod) * 100
    y = np.concatenate([[0], np.cumsum(prod) / prod.sum() * 100])
    k10 = int(np.ceil(0.1 * len(prod)))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 100], [0, 100], color=AXIS, lw=1)
    ax.plot(x, y, color=SERIES[0])
    ax.scatter([x[k10]], [y[k10]], s=36, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"Top 10% of programs\ntrained {y[k10]:.0f}% of chairs", (x[k10], y[k10]),
                xytext=(18, 50), textcoords="data", color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(62, 52, "equal production", color=MUTED, fontsize=8, rotation=38)
    ax.set_xlabel("Residency programs, ranked by chairs produced (cumulative %)")
    ax.set_ylabel("Chairs trained (cumulative %)")
    ax.set_title("Chair production is concentrated")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    save(fig, "fig1_lorenz_chair_production")


def fig2_steps():
    s = pd.read_csv(TABLES / "placement_steps_chair.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.9))
    jitter = np.random.default_rng(0).normal(0, 0.6, (2, len(s)))
    ax.plot([0, 100], [0, 100], color=AXIS, lw=1)
    ax.scatter(s.source_pct + jitter[0], s.target_pct + jitter[1], s=14, color=SERIES[0],
               alpha=0.55, edgecolor="none")
    down = (s.step_pct < 0).mean() * 100
    ax.text(3, 80, f"Above the line: moved up\nBelow: moved down ({down:.0f}% of\nexternal chair appointments)",
            color=INK2, fontsize=8, va="top")
    ax.set_xlabel("Training program rank (percentile, 100 = top)")
    ax.set_ylabel("Program where chair was held (percentile)")
    ax.set_title("Chairs move down the placement hierarchy")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    save(fig, "fig2_placement_steps")


def fig3_top_programs(n=25):
    r = pd.read_csv(TABLES / "program_ranking_chair.csv")
    r = r[r.leaders_produced >= 3].nsmallest(n, "boot_rank_median").iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.22 * n + 1))
    yy = np.arange(len(r))
    ax.hlines(yy, r.boot_rank_p05, r.boot_rank_p95, color="#86b6ef", lw=2)
    ax.scatter(r.boot_rank_median, yy, s=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(yy, [f"{p}  ({int(k)})" for p, k in zip(r.program, r.leaders_produced)], fontsize=7.5, color=INK2)
    ax.set_xlabel("Bootstrap median rank (dot) with 90% interval\nprograms training ≥3 chairs; (n) = chairs trained")
    ax.set_title("Top programs in the chair placement hierarchy")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, None)
    save(fig, "fig3_top_programs_mvr")


def fig4_eras():
    e = pd.read_csv(TABLES / "hierarchy_by_era_chair.csv")
    panels = [("self_hire_rate", "Chairs appointed at own\nresidency program"),
              ("top10_share", "Chairs trained by the\n10 leading programs"),
              ("female_share", "Chairs who are women")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
    x = np.arange(len(e))
    for ax, (col, title) in zip(axes, panels):
        v = e[col] * 100
        ax.plot(x, v, color=SERIES[0], marker="o", ms=5, mec=SURFACE, mew=1.5)
        ax.text(x[-1], v.iloc[-1], f" {v.iloc[-1]:.0f}%", va="center", color=INK2, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x, e.era.str.replace("-present", "+"), fontsize=7, rotation=0)
        ax.set_ylim(0, max(10, v.max() * 1.25))
        ax.set_xlim(-0.3, len(e) - 0.4)
    axes[0].set_ylabel("% of chair appointments")
    fig.supxlabel("Era of chair appointment", fontsize=8, color=INK2)
    save(fig, "fig4_era_trends")


def fig5_incidence():
    df = pd.read_csv(ROOT / "data" / "processed" / "inheritance_cohort.csv")
    df = df[df.has_chair_link == 1]
    cats = pd.cut(df.leader_makers_prior, [-1, 0, 2, 1e9], labels=["0", "1-2", "3+"])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for (label, g), color in zip(df.groupby(cats, observed=True), SERIES):
        sf = SurvfuncRight(g.time, g.event)
        t = np.concatenate([[0], sf.surv_times]); ci = np.concatenate([[0], 1 - sf.surv_prob]) * 100
        ax.step(t, ci, where="post", color=color, label=f"{label} (n={len(g):,})")
        ax.text(min(t.max(), 40) + 0.5, ci[t <= 40][-1], label, color=INK2, fontsize=8, va="center")
    ax.set_xlim(0, 43)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    ax.set_xlabel("Years after residency graduation")
    ax.set_ylabel("Appointed chair (cumulative %)")
    ax.set_title("Trainees of leader-makers become chairs more often")
    ax.legend(loc="upper left", fontsize=7.5, labelcolor=INK2, alignment="left",
              title="Earlier trainees of chair/PD\nwho became chairs", title_fontsize=7.5)
    save(fig, "fig5_chair_incidence_by_leader_maker")


def fig6_forest():
    m = pd.read_csv(TABLES / "inheritance_cox_models.csv")
    names = {"log_leader_makers_prior": "per log(1+n)", "log_chair_makers_prior": "chair only, per log(1+n)",
             "log_pd_makers_prior": "PD only, per log(1+n)", "maker_3p": "3+ vs 0"}
    m = m[m.term.isin(names) & m.model.str.startswith(("M", "S", "Ec"))].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(m) + 1.2))
    yy = np.arange(len(m))
    ax.axvline(1, color=AXIS, lw=1)
    ax.hlines(yy, m.CI_low, m.CI_high, color="#86b6ef", lw=2)
    ax.scatter(m.HR, yy, s=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    labels = [f"{mod}  [{names[t]}]" for mod, t in zip(m.model, m.term)]
    ax.set_yticks(yy, labels, fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.4, max(3.5, m.CI_high.max() * 1.9))
    for y_, hr, lo, hi in zip(yy, m.HR, m.CI_low, m.CI_high):
        ax.text(hi * 1.05, y_, f"{hr:.2f} ({lo:.2f}-{hi:.2f})", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("Hazard ratio for chair appointment\n(n = earlier trainees of the chair/PD who became chairs)")
    ax.set_title("Leader-maker effect across models")
    ax.grid(axis="y", visible=False)
    save(fig, "fig6_forest_leader_maker")


if __name__ == "__main__":
    for f in (fig1_lorenz, fig2_steps, fig3_top_programs, fig4_eras, fig5_incidence, fig6_forest):
        f()
        print("done", f.__name__)
