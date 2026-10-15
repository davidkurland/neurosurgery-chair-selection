"""Main and supplementary figures (PNG 300 dpi + PDF) -> outputs/figures/v2/

Fig 1  Where chairs came from: flow map of chair appointments by era
Fig 2  The changing path to the chair: age, time since residency, tenure, succession
Fig 3  What did not change: concentration, self-hiring, women vs expectation
Fig 4  Chairmakers: chairs trained by the most prolific mentors
Fig S2 Chairmaker family trees (two generations)
Fig S3 Mentor leader-making: sensitivity of the >=3 association
Fig S4 Self-hiring by program against the spread expected from a common rate
Placement order: programs furthest upstream in chair placement (figS_placement_order)
(Supplemental ranking and incidence figures come from code/05_figures.py.)

Run after the table-building scripts 03-08, 10-12, 15, 16 and 18 (or 22):
    python code/09_figures_v2.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
EXT = ROOT / "data" / "external"
FIGS = ROOT / "outputs" / "figures" / "v2"
FIGS.mkdir(parents=True, exist_ok=True)

# figure palette (light background)
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal, >=2:1 on light
BLUE_LIGHT, BLUE_DARK = "#9ec5f4", "#1c5cab"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, LAND = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.linewidth": 2, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def clean_legend(ax, **kw):
    """Legend swatches without markers: the white marker edge otherwise splits the
    line and makes a solid series look dashed."""
    hs, ls = ax.get_legend_handles_labels()
    hs = [Line2D([], [], color=h.get_color(), lw=h.get_linewidth(), ls=h.get_linestyle())
          for h in hs]
    return ax.legend(hs, ls, **kw)


def panel_letter(ax, letter):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------ map ----
def albers(lon, lat, lon0=-96.0, lat0=37.5, p1=29.5, p2=45.5):
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    lon0, lat0, p1, p2 = map(np.radians, (lon0, lat0, p1, p2))
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = np.sqrt(c - 2 * n * np.sin(lat0)) / n
    rho = np.sqrt(c - 2 * n * np.sin(lat)) / n
    theta = n * (lon - lon0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def polygons(geojson, keep):
    feats = json.loads(Path(geojson).read_text())["features"]
    for f in feats:
        if not keep(f["properties"]):
            continue
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0])
            if ring[:, 1].max() < 23 or ring[:, 0].min() > -50 or ring[:, 0].max() < -140:
                continue
            yield ring


def draw_basemap(ax):
    for ring in polygons(EXT / "ne_50m_admin_0_countries.geojson",
                         lambda p: p.get("ADM0_A3") in ("MEX", "CAN", "USA")):
        x, y = albers(ring[:, 0], ring[:, 1])
        ax.fill(x, y, color=LAND, lw=0, zorder=0)
    for ring in polygons(EXT / "ne_50m_admin_1_states_provinces.geojson",
                         lambda p: p.get("adm0_a3") in ("USA", "CAN") and p.get("name") not in ("Alaska", "Hawaii")):
        x, y = albers(ring[:, 0], ring[:, 1])
        ax.plot(x, y, color="#d6d4cc", lw=0.35, zorder=1)
    _, ybot = albers([-81], [22.5])   # room below Miami so bowed arcs stay inside
    xx, _ = albers([-125, -64], [40, 40])
    _, ytop = albers([-96], [54.5])
    ax.set_xlim(xx[0] - 0.01, xx[1] + 0.01)
    ax.set_ylim(ybot[0], ytop[0])
    ax.set_aspect("equal")
    ax.axis("off")


def arc(x0, y0, x1, y1, bend=0.12, n=24):
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bend, my + dx * bend
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array([x0, y0]) + 2 * (1 - t) * t * np.array([cx, cy]) + t ** 2 * np.array([x1, y1])


def fig1_map():
    e = pd.read_csv(TABLES / "placement_edges_chair.csv")
    inst = pd.read_csv(ROOT / "data" / "processed" / "institutions.csv")
    na = inst[inst.country.isin(["United States", "Canada"])]
    e = e[e.source.isin(na.institution_id) & e.target.isin(na.institution_id)]
    # panel counts use ALL North American appointments (matching Tables 1-2); arcs are
    # drawn only for programs with coordinates inside the mainland extent
    e_all = e.copy()
    e = e.dropna(subset=["source_lat", "source_lon", "target_lat", "target_lon"])
    in_box = lambda lat, lon: lat.between(24, 56) & lon.between(-130, -60)
    e = e[in_box(e.source_lat, e.source_lon) & in_box(e.target_lat, e.target_lon)]
    print(f"map: {len(e_all) - len(e)} of {len(e_all)} appointments not drawn (no coordinates or off-map)")
    eras = [("pre-1970", "Before 1970"), ("1970-1989", "1970–1989"), ("1990-2009", "1990–2009"), ("2010-present", "2010–2026")]
    succ_fig1 = pd.read_csv(TABLES / "lifecycle_successions.csv")
    succ_fig1["era4"] = pd.cut(succ_fig1.start, [0, 1969, 1989, 2009, 2100],
                               labels=["pre-1970", "1970-1989", "1990-2009", "2010-present"]).astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    px, py = albers(na.lon, na.lat)
    for ax, (key, label) in zip(axes.flat, eras):
        draw_basemap(ax)
        sub = e[e.era == key]
        ext, own = sub[~sub.self_hire], sub[sub.self_hire]
        ax.scatter(px, py, s=2, color=MUTED, alpha=0.5, lw=0, zorder=2)
        for r in ext.itertuples():
            x0, y0 = albers(r.source_lon, r.source_lat)
            x1, y1 = albers(r.target_lon, r.target_lat)
            pts = arc(float(x0), float(y0), float(x1), float(y1))
            ax.plot(pts[:, 0], pts[:, 1], color=BLUE, lw=0.55, alpha=0.35, zorder=3, solid_capstyle="round")
        prod = sub.groupby(["source", "source_lat", "source_lon"]).size().reset_index(name="n")
        sx, sy = albers(prod.source_lon, prod.source_lat)
        ax.scatter(sx, sy, s=6 + prod.n * 9, color=BLUE_DARK, alpha=0.85, lw=0.6, edgecolor=SURFACE, zorder=4)
        ho = own.groupby(["target", "target_lat", "target_lon"]).size().reset_index(name="n")
        hx, hy = albers(ho.target_lon, ho.target_lat)
        ax.scatter(hx, hy, s=14 + ho.n * 14, facecolor="none", edgecolor=ORANGE, lw=1.1, zorder=5)
        ax.set_title(f"{label}", loc="left", fontsize=9.5)
        full = e_all[e_all.era == key]
        # own-graduate share among successions (founding chairs excluded), matching Table 2 and the text
        sh = succ_fig1[succ_fig1.era4 == key]
        ax.text(0.0, 0.985, f"{len(full)} appointments · {100 * sh.own_graduate.mean():.0f}% of successions filled by an own graduate",
                transform=ax.transAxes, fontsize=7.2, color=INK2, va="top")
    handles = [Line2D([], [], color=BLUE, lw=1.2, alpha=0.7, label="Chair moved from training program to another program"),
               Line2D([], [], marker="o", ls="", markersize=7, color=BLUE_DARK, label="Training program (size = chairs trained)"),
               Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor="none", markeredgecolor=ORANGE,
                      markeredgewidth=1.2, label="Chair appointed at own training program")]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=7.5, bbox_to_anchor=(0.5, -0.05), labelcolor=INK2)
    fig.suptitle("Where neurosurgery's chairs trained and where they led", x=0.02, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save(fig, "fig1_chair_flow_map")


# ------------------------------------------------------------ lifecycle ----
def fig2_lifecycle():
    pts = pd.read_csv(TABLES / "lifecycle_first_chair_points.csv")
    curves = pd.read_csv(TABLES / "lifecycle_tenure_km_curves.csv")
    ten = pd.read_csv(TABLES / "lifecycle_tenure_by_era.csv")
    succ = pd.read_csv(TABLES / "lifecycle_succession_by_era.csv")
    succ = succ[succ.era != "All"]
    wf = pd.read_csv(TABLES / "v06_workforce_by_year.csv")
    pctd = pd.read_csv(TABLES / "v06_chair_percentile_by_decade.csv")
    kmb = pd.read_csv(TABLES / "v06_tenure_km_age45_55.csv")
    pubp = pd.read_csv(TABLES / "v08_chair_pub_percentile_by_decade.csv")
    jit = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.9))

    # A, B: chairs against the active workforce of the same year
    for ax, col, pre, title, ylab, letter in [
            (axes[0, 0], "age", "age", "Age at first chair appointment", "Age (years)", "A"),
            (axes[0, 1], "years_post_grad", "exp", "Time from residency to first chair",
             "Years after residency graduation", "B")]:
        w = wf.dropna(subset=[f"{pre}_p50"])
        w = w[w.year >= 1920]
        ax.fill_between(w.year, w[f"{pre}_p25"], w[f"{pre}_p75"], color=LAND, lw=0, zorder=0)
        ax.plot(w.year, w[f"{pre}_p50"], color=MUTED, lw=1.3, ls=(0, (4, 2)), zorder=1)
        d = pts.dropna(subset=[col])
        d = d[d.start >= 1920]
        ax.scatter(d.start + jit.normal(0, 0.6, len(d)), d[col], s=6, color=BLUE_LIGHT, alpha=0.55, lw=0, zorder=2)
        m = d.groupby((d.start // 10 * 10).astype(int))[col].median()
        m = m[d.groupby((d.start // 10 * 10).astype(int)).size() >= 10]
        ax.plot(m.index + 5, m.values, color=BLUE_DARK, marker="o", ms=4.5, mec=SURFACE, mew=1.2, zorder=3)
        for yr in (m.index[0], m.index[-1]):
            ax.text(yr + 5, m[yr] + (3 if col == "age" else 2.5), f"{yr}s: {m[yr]:.0f}", ha="center",
                    color=INK2, fontsize=7.5, zorder=4)
        anchor = 1948
        y_at = float(np.interp(anchor, w.year, w[f"{pre}_p50"]))
        ax.annotate("active workforce\n(median, IQR)", xy=(anchor, y_at), xycoords="data",
                    xytext=(0.03, 0.96), textcoords="axes fraction", ha="left", va="top",
                    fontsize=7, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                    connectionstyle="angle3,angleA=0,angleB=90"))
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Year of appointment")
        panel_letter(ax, letter)

    # C: where new chairs sit inside that workforce
    ax = axes[0, 2]
    ax.axhline(50, color=AXIS, lw=1.2, ls=(0, (4, 2)))
    ax.text(pctd.decade.iloc[0] + 3, 52, "midpoint of the comparison group", fontsize=7, color=MUTED)
    ax.plot(pctd.decade + 5, 100 * pctd.pct_experience, color=BLUE_DARK, marker="o", ms=4.5,
            mec=SURFACE, mew=1.2, label="Seniority: years since graduation")
    ax.plot(pctd.decade + 5, 100 * pctd.pct_age, color=BLUE_LIGHT, marker="o", ms=3.8, lw=1.5,
            mec=SURFACE, mew=1.0, label="Seniority: age")
    ax.plot(pubp.decade + 5, 100 * pubp.pct_publications, color=ORANGE, marker="o", ms=4.5,
            mec=SURFACE, mew=1.2, label="Publications, vs own cohort")
    ax.set_ylim(0, 104)
    ax.set_title("Where new chairs sit among their peers")
    ax.set_ylabel("Percentile")
    ax.set_xlabel("Decade of appointment")
    clean_legend(ax, loc="lower right", fontsize=6.5, labelcolor=INK2)
    panel_letter(ax, "C")

    # D: tenure by era
    ax = axes[1, 0]
    eras = list(dict.fromkeys(curves.era))
    for color, e in zip(RAMP, eras):
        c = curves[curves.era == e]
        c = c[c.years <= 30]
        med = ten.loc[ten.era == e, "km_median_years"].iloc[0]
        lab = e.replace("-", "\u2013").replace("<1950", "Before 1950")
        ax.step(c.years, 100 * c.surv, where="post", color=color, lw=1.8,
                label=f"{lab}: {med:.0f} y" if pd.notna(med) else lab)
    ax.legend(loc="upper right", fontsize=6.5, labelcolor=INK2, title="Appointed (median tenure)",
              title_fontsize=6.5, alignment="left")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_title("Chair tenure, all appointments")
    ax.set_ylabel("Still serving as chair (%)")
    ax.set_xlabel("Years since appointment")
    panel_letter(ax, "D")

    # E: tenure within a fixed age band at appointment
    ax = axes[1, 1]
    for color, e in zip(RAMP, eras):
        c = kmb[kmb.era == e]
        if c.empty:
            continue
        c = c[c.years <= 30]
        # tolerance: S(t) = 1/2 exactly is stored as 0.5000000000000001 (pre-1950, t = 13)
        below = c[c.surv <= 0.5 + 1e-9]
        med = below.years.iloc[0] if len(below) else np.nan
        lab = e.replace("-", "\u2013").replace("<1950", "Before 1950")
        ax.step(c.years, 100 * c.surv, where="post", color=color, lw=1.8,
                label=f"{lab}: {med:.0f} y" if pd.notna(med) else lab)
    ax.legend(loc="upper right", fontsize=6.5, labelcolor=INK2, title="Appointed at 45\u201355 (median)",
              title_fontsize=6.5, alignment="left")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_title("Chair tenure at a fixed age of entry")
    ax.set_ylabel("Still serving as chair (%)")
    ax.set_xlabel("Years since appointment")
    panel_letter(ax, "E")

    # F: who succeeds
    ax = axes[1, 2]
    x = np.arange(len(succ))
    ax.plot(x, succ.own_graduate_pct, color=ORANGE, marker="o", ms=4.5, mec=SURFACE, mew=1.2,
            label="New chair graduated from the program")
    ax.plot(x, succ.trained_by_predecessor_pct, color=BLUE, marker="o", ms=4.5, mec=SURFACE, mew=1.2,
            label="New chair trained by the outgoing chair")
    chance = 100 * succ.chance_national_expected_n / succ.transitions
    ax.plot(x, chance, color=AXIS, marker="o", ms=3.5, lw=1.5, label="\u2026expected by chance")
    pct = lambda v: f"{int(np.floor(v + 0.5))}%"
    for series, col in ((succ.own_graduate_pct, INK2), (succ.trained_by_predecessor_pct, INK2)):
        ax.text(x[0] - 0.15, series.iloc[0], pct(series.iloc[0]), ha="right", va="center", fontsize=7.5, color=col)
        ax.text(x[-1] + 0.15, series.iloc[-1], pct(series.iloc[-1]), ha="left", va="center", fontsize=7.5, color=col)
    ax.set_xticks(x, [e.replace("-", "\u2013") for e in succ.era], fontsize=6.6)
    ax.set_xlim(-0.7, len(succ) - 0.3)
    ax.set_ylim(0, max(50, succ.own_graduate_pct.max() * 1.15))
    ax.set_title("Who succeeds the outgoing chair")
    ax.set_ylabel("Chair successions (%)")
    ax.set_xlabel("Era of appointment")
    clean_legend(ax, loc="upper right", fontsize=6.5, labelcolor=INK2)
    panel_letter(ax, "F")
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    save(fig, "fig2_chair_lifecycle")


# ------------------------------------------------------- what persisted ----
def fig3_unchanged():
    conc = pd.read_csv(TABLES / "robustness_concentration_vs_size.csv")
    mentors = pd.read_csv(TABLES / "v03_mentor_concentration_vs_null_4era.csv")
    lag = pd.read_csv(TABLES / "v05_women_denominators.csv")
    order = ["pre-1970", "1970-1989", "1990-2009", "2010-present"]
    labels = ["<1970", "1970–\n1989", "1990–\n2009", "2010–\n2026"]
    conc = conc.set_index("era").loc[order]
    mentors = mentors.set_index("era").loc[order]
    # self-hiring among successions (founding chairs excluded), vs program-size expectation
    sh = pd.read_csv(TABLES / "v04_selfhire_successions_4era.csv").set_index("era").loc[order]
    sh_obs = sh.own_graduate_pct.to_numpy()
    sh_null = sh.expected_pct.to_numpy()
    # women: observed share, and expected share under the era-specific lag (0 before 1970)
    es = lag[(lag.denominator == "all") & (lag.lag == "era-specific")].set_index("era")
    w_obs = [0, 0] + [100 * es.loc[e, "women_observed"] / es.loc[e, "appointments"] for e in order[2:]]
    # v05_women_denominators.csv has no pre-1990 rows: the pre-1970 and 1970-1989 expected values (0 and 0.2%)
    # are fixed here
    w_exp = [0, 0.2] + [100 * es.loc[e, "women_expected"] / es.loc[e, "appointments"] for e in order[2:]]
    pdw = pd.read_csv(TABLES / "v06_women_pd_4era.csv").set_index("era").loc[order]
    pd_obs = (100 * pdw.women_observed / pdw.pd_appointments).to_numpy()
    pd_exp = (100 * pdw.women_expected / pdw.pd_appointments).to_numpy()
    panels = [
        ("Successors who graduated\nfrom the program", sh_obs, sh_null, "Expected from program size"),
        ("Chairs trained by the 10\nmost prolific programs", 100 * conc.top10_share_observed, 100 * conc.top10_share_size_only,
         "Expected from program size"),
        ("Chairs trained by the 10\nmost prolific mentors", 100 * mentors.top10_share_observed,
         100 * mentors.top10_share_null, "Expected from mentors' trainee counts"),
        ("Appointments held by women:\nchairs and program directors", w_obs, w_exp,
         "Expected from the graduate pipeline"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.9))
    x = np.arange(4)
    r0 = lambda v: int(np.floor(v + 0.5))   # round half up, so 20.5 prints as 21 like the text
    ylabels = {"A": "% of successions", "B": "% of chairs trained", "C": "% of chairs trained", "D": "% of appointments"}
    for ax, (title, obs, exp, exp_label), letter in zip(axes, panels, "ABCD"):
        ax.set_ylabel(ylabels[letter])
        obs, exp = np.asarray(obs, float), np.asarray(exp, float)
        ax.plot(x, exp, color=AXIS, marker="o", ms=3.5, lw=1.5, label=exp_label)
        ax.plot(x, obs, color=BLUE, marker="o", ms=4.5, mec=SURFACE, mew=1.2,
                label="Chairs, observed" if letter == "D" else "Observed")
        if letter == "D":
            ax.plot(x, pd_exp, color=AXIS, marker="s", ms=3.0, lw=1.2, ls=(0, (3, 2)),
                    label="PDs, expected")
            ax.plot(x, pd_obs, color=ORANGE, marker="s", ms=4.0, mec=SURFACE, mew=1.1,
                    label="PDs, observed")
            obs_max = max(obs.max(), exp.max(), pd_obs.max(), pd_exp.max())
            ax.set_ylim(0, obs_max * 1.6 + 1)
            ax.set_yticks(np.arange(0, obs_max * 1.6 + 1, 5))
            ax.text(x[-1] + 0.12, pd_obs[-1] + 0.45, f"{r0(pd_obs[-1])}% PDs", va="bottom",
                    fontsize=7.5, color=ORANGE)
            ax.text(x[-1] + 0.12, obs[-1] - 0.45, f"{r0(obs[-1])}% chairs", va="top",
                    fontsize=7.5, color=BLUE)
        span = max(obs.max(), exp.max()) * 1.25 + 1
        if letter == "D":
            pass   # panel D labels are placed above
        elif abs(obs[-1] - exp[-1]) < 0.07 * span:  # labels would collide: stack them
            ax.text(x[-1] + 0.12, max(obs[-1], exp[-1]) + 0.035 * span, f"{r0(obs[-1])}% observed", va="center",
                    fontsize=7, color=INK2)
            ax.text(x[-1] + 0.12, min(obs[-1], exp[-1]) - 0.035 * span, f"{r0(exp[-1])}% expected", va="center",
                    fontsize=7, color=MUTED)
        else:
            ax.text(x[-1] + 0.12, obs[-1], f"{r0(obs[-1])}%", va="center", fontsize=7.5, color=INK2)
            ax.text(x[-1] + 0.12, exp[-1], f"{r0(exp[-1])}%", va="center", fontsize=7.5, color=MUTED)
        ax.set_xticks(x, labels, fontsize=6.8)
        ax.set_xlim(-0.3, 3.7)
        if letter != "D":
            ax.set_ylim(0, max(obs.max(), exp.max()) * 1.5 + 1)
        ax.set_title(title, fontsize=8.8)
        clean_legend(ax, loc="upper left", fontsize=6.0 if letter == "D" else 6.5,
                     labelcolor=INK2, handlelength=1.9, borderaxespad=0.3)
        panel_letter(ax, letter)
    fig.supxlabel("Era of appointment", fontsize=8, color=INK2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_what_persisted")


# ------------------------------------------------------------ chairmakers ----
def fig4_chairmakers(n=15):
    km = pd.read_csv(TABLES / "lifecycle_chairmakers.csv").head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.27 * n + 1.1))
    y = np.arange(len(km))
    ax.barh(y, km.chairs_trained, color=BLUE, height=0.62, edgecolor=SURFACE, linewidth=1.2,
            label="Chairs trained directly")
    ax.barh(y, km.second_generation_chairs, left=km.chairs_trained, color=BLUE_LIGHT, height=0.62,
            edgecolor=SURFACE, linewidth=1.2, label="Chairs trained by those chairs")
    for yi, r in zip(y, km.itertuples()):
        ax.text(r.chairs_trained + r.second_generation_chairs + 0.6, yi,
                f"{r.chairs_trained} + {r.second_generation_chairs}", va="center", fontsize=7.2, color=INK2)
    names = [f"{r.name}  ({int(r.first_trainee_grad)}–{int(r.last_trainee_grad)})" for r in km.itertuples()]
    ax.set_yticks(y, names, fontsize=7.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Department chairs (US and Canadian programs and beyond)")
    ax.set_title("The chairmakers: neurosurgeons who trained the most chairs")
    ax.legend(loc="lower right", fontsize=7.2, labelcolor=INK2)
    fig.tight_layout()
    fig.text(0.01, -0.01, "Years in parentheses: graduation years of the mentor's trainees.", fontsize=6.8,
             color=MUTED, va="top")
    save(fig, "fig4_chairmakers")


def figS2_trees(k=4):
    te = pd.read_csv(TABLES / "lifecycle_chairmaker_tree_edges.csv")
    roots = list(dict.fromkeys(te.root))[:k]
    gen1_sizes = [len(te[(te.root == r) & (te.generation == 1)]) for r in roots]
    fig, axes = plt.subplots(1, k, figsize=(7.4, 0.2 * max(gen1_sizes) + 1.2))
    for ax, root in zip(axes, roots):
        g1 = te[(te.root == root) & (te.generation == 1)].sort_values("child_first_chair")
        g2 = te[(te.root == root) & (te.generation == 2)]
        # second generation excludes chairs already trained directly by the root (as in Table 3)
        g2 = g2[~g2.child.isin(set(g1.child) | {root})]
        kids = g2.groupby("parent").child.nunique()
        n = len(g1)
        ys = np.linspace(n - 1, 0, n) if n > 1 else np.array([0.0])
        ymid = (n - 1) / 2
        for yv in ys:
            ax.plot([0, 0.35, 0.35, 0.55], [ymid, ymid, yv, yv], color=AXIS, lw=0.7, solid_joinstyle="round")
        ax.scatter([0], [ymid], s=40, color=BLUE_DARK, zorder=3)
        ax.scatter(np.full(n, 0.55), ys, s=14, color=BLUE, zorder=3)
        for yv, r in zip(ys, g1.itertuples()):
            c = kids.get(r.child, 0)
            yr = f"{int(r.child_first_chair)}" if pd.notna(r.child_first_chair) else ""
            ax.text(0.62, yv, f"{r.child_name} {yr}" + (f"  (+{c})" if c else ""), va="center", fontsize=5.6, color=INK2)
        ax.set_title(f"{g1.root_name.iloc[0]}\n{n} chairs; {g2.child.nunique()} more in the next generation",
                     fontsize=7.2, loc="left")
        ax.set_xlim(-0.05, 3.2)
        ax.set_ylim(-1, max(gen1_sizes))
        ax.axis("off")
    fig.text(0.01, 0.005, "Each name is a chair trained by the mentor (year of first chair appointment); "
             "(+n) = chairs that person trained in turn, excluding the mentor's own direct trainees; a chair trained by two of them is counted under each.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, "figS2_chairmaker_trees")


def figS4_selfhire_by_program():
    """Every program ranked by how often it hired its own graduate, against the
    spread a single common rate would produce."""
    prog = pd.read_csv(TABLES / "v06_selfhire_by_program.csv")
    big = prog[prog.successions >= 3].copy()
    n = big.successions.to_numpy(int)
    p = big.own_graduate.sum() / big.successions.sum()   # pooled rate over the programs with >=3 successions
    obs = np.sort(100 * big.own_graduate.to_numpy() / n)[::-1]
    rng = np.random.default_rng(9)
    sims = np.sort(100 * rng.binomial(n, p, size=(4000, len(n))) / n, axis=1)[:, ::-1]
    lo, hi = np.percentile(sims, 5, axis=0), np.percentile(sims, 95, axis=0)
    x = np.arange(1, len(obs) + 1)
    n_zero = int((big.own_graduate == 0).sum())
    n_third = int((obs >= 100 / 3 - 0.5).sum())

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.fill_between(x, lo, hi, color=LAND, lw=0, step="mid", zorder=0)
    ax.bar(x, obs, width=0.92, color=BLUE, lw=0, zorder=2, label="Observed, one bar per program")
    for edge in (lo, hi):
        ax.step(x, edge, where="mid", color=MUTED, lw=1.0, zorder=5)
    ax.plot([], [], color=MUTED, lw=1.0,
            label=f"Range expected (5th\u201395th percentile) if every\nprogram hired its own at the pooled {100 * p:.0f}%")
    ax.axhline(100 * p, color=ORANGE, lw=1.6, ls=(0, (4, 2)), zorder=4)
    ax.text(len(obs), 100 * p + 3, f"pooled {100 * p:.0f}%", ha="right", fontsize=8, color=ORANGE)

    first_zero = int(np.argmax(obs == 0)) + 1
    ax.annotate("", xy=(first_zero, 7), xytext=(len(obs), 7),
                arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3", color=INK2, lw=0.9))
    ax.text((first_zero + len(obs)) / 2, 10.5,
            f"{n_zero} programs ({100 * n_zero / len(obs):.0f}%) never appointed an own graduate",
            ha="center", fontsize=7.5, color=INK2)
    ax.annotate("", xy=(1, obs[0] + 3), xytext=(n_third, obs[0] + 3),
                arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3", color=INK2, lw=0.9))
    ax.text(n_third + 2, obs[0] + 3,
            f"{n_third} programs filled a third or more of their\nsuccessions with their own graduates",
            ha="left", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0.3, len(obs) + 0.7)
    ax.set_ylim(0, 112)
    ax.set_xlabel(f"The {len(obs)} programs with \u22653 chair successions since 1970, "
                  "ranked by self-hiring")
    ax.set_ylabel("Successions filled by an\nown graduate (%)")
    ax.set_title("Self-hiring is concentrated in some departments, absent in others")
    ax.legend(loc="upper right", fontsize=7.5, labelcolor=INK2, handlelength=1.6,
              bbox_to_anchor=(1.0, 0.86))
    # p value: overdispersion test against a common rate (overdispersion_p, script 16_v06_strengthen.py)
    ax.text(len(obs), 47, "spread wider than a common rate allows, p < 0.001",
            ha="right", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, "figS4_selfhire_by_program")


def figS3_inheritance_forest():
    main = pd.read_csv(TABLES / "inheritance_cox_models.csv")
    rob = pd.read_csv(TABLES / "robustness_inheritance_models.csv")
    fol = pd.read_csv(TABLES / "followup_program_era_models.csv")
    pick = [
        (main, "M2c within-program, categorical", "Primary: within program, linear time trend"),
        (fol, "F3 categorical, cubic trend", "Cubic time trend"),
        (fol, "F2 categorical + recent program rate", "+ program's recent chair production"),
        (rob, "R4a strata program x decade, categorical", "Within program × decade"),
        (main, "Ec within-program categorical, grads <1970", "Graduates before 1970"),
        (main, "Ec within-program categorical, grads 1970-1989", "Graduates 1970–1989"),
        (main, "Ec within-program categorical, grads 1990+", "Graduates 1990 onward"),
        (rob, "R4b 15-y window categorical, grads <1970", "15-y follow-up, graduates before 1970"),
        (rob, "R4b 15-y window categorical, grads 1970-1989", "15-y follow-up, graduates 1970–1989"),
        (rob, "R4b 15-y window categorical, grads 1990-2010", "15-y follow-up, graduates 1990–2010"),
    ]
    rows = []
    for df, model, label in pick:
        r = df[(df.model == model) & (df.term == "maker_3p")]
        if len(r):
            rows.append((label, *r[["HR", "CI_low", "CI_high"]].iloc[0]))
    d = pd.DataFrame(rows, columns=["label", "HR", "lo", "hi"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(d) + 1.1))
    y = np.arange(len(d))
    ax.axvline(1, color=AXIS, lw=1)
    ax.hlines(y, d.lo, d.hi, color=BLUE_LIGHT, lw=2.2)
    ax.scatter(d.HR, y, s=30, color=BLUE, edgecolor=SURFACE, linewidth=1.4, zorder=3)
    for yi, r in zip(y, d.itertuples()):
        ax.text(r.hi * 1.06, yi, f"{r.HR:.2f} ({r.lo:.2f}–{r.hi:.2f})", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, d.label, fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.22, 8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hazard ratio for chair appointment\n(mentor with ≥3 vs 0 earlier trainees who became chairs)")
    ax.set_title("Mentor leader-making: sensitivity of the association")
    fig.tight_layout()
    save(fig, "figS3_inheritance_sensitivity")


def figS_placement_order(n=25):
    """Programs furthest upstream in the direction of chair placement (minimum-violation order),
    with bootstrap rank intervals."""
    r = pd.read_csv(TABLES / "program_ranking_chair.csv")
    r = r[r.leaders_produced >= 3].nsmallest(n, "boot_rank_median").iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.22 * n + 1))
    yy = np.arange(len(r))
    ax.hlines(yy, r.boot_rank_p05, r.boot_rank_p95, color=RAMP[0], lw=2)
    ax.scatter(r.boot_rank_median, yy, s=30, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(yy, [f"{p}  ({int(k)})" for p, k in zip(r.program, r.leaders_produced)], fontsize=7.5, color=INK2)
    ax.set_xlabel("Position in the placement order (1 = furthest upstream)\n"
                  "bootstrap median (dot) with 90% interval\n"
                  "programs training ≥3 chairs; (n) = chairs trained")
    ax.set_title("Programs furthest upstream in chair placement")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, None)
    save(fig, "figS_placement_order")


if __name__ == "__main__":
    for f in (fig1_map, fig2_lifecycle, fig3_unchanged, fig4_chairmakers, figS2_trees, figS3_inheritance_forest,
              figS4_selfhire_by_program, figS_placement_order):
        f()
