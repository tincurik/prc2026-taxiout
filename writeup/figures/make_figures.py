# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the write-up figures from the small tables in ../data/ (no model is run here).

    python make_figures.py            # writes PNG (200 dpi) and SVG next to this file

Style: one surface, text in ink tokens (never in the data colour), hairline grid, 4 px rounded bar ends are
approximated by flat bars with a 2 px surface gap. Categorical colours are slots 1-3 of a validated palette
(blue, orange, aqua; worst adjacent colour-blind separation dE 9.2 in OKLab x100), plus neutral grey for "removed".
"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FAMILY_COLOR = {"Label structure": BLUE, "Models and features": ORANGE, "Open data": AQUA, "Removed": MUTED}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "taxi-out",
})


def header(fig, title, subtitle, y=0.975):
    fig.text(0.02, y, title, fontsize=15, fontweight="bold", ha="left", va="top", color=INK)
    sub = textwrap.fill(subtitle, 125)
    fig.text(0.02, y - 0.05, sub, fontsize=10, ha="left", va="top", color=INK2, linespacing=1.4)


def footer(fig, text):
    fig.text(0.02, 0.012, textwrap.fill(text, 170), fontsize=8, ha="left", va="bottom", color=MUTED, linespacing=1.4)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)
    print("wrote", name)


def style_axis(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- 1. the progress chart (hero)
def fig_progress():
    d = pd.read_csv(DATA / "progress.csv")
    n = len(d)
    rows = n + 2
    fig, ax = plt.subplots(figsize=(10.4, 8.4))
    fig.subplots_adjust(left=0.40, right=0.97, top=0.86, bottom=0.11)
    ys = np.arange(rows)[::-1]  # top row first
    base = d.cv_from.iloc[0]
    final = d.cv_to.iloc[-1]
    ax.set_ylim(-0.7, rows - 0.3)
    ax.set_xlim(270, 735)
    style_axis(ax, "x")

    labels = ["Airport-mean baseline"] + d.stage.tolist() + ["Final model"]
    # baseline and final: dots with a surface ring
    for y, v in ((ys[0], base), (ys[-1], final)):
        ax.plot([v], [y], "o", ms=10, color=INK, mec=SURFACE, mew=2, zorder=5)
        ax.text(v + 9, y, f"{v:.2f} s", va="center", ha="left", fontsize=10, fontweight="bold", color=INK)
    prev_left = None
    for i, r in d.iterrows():
        y = ys[i + 1]
        lo, hi = sorted((r.cv_from, r.cv_to))
        col = FAMILY_COLOR[r.family]
        h = 0.56
        ax.add_patch(FancyBboxPatch((lo, y - h / 2), max(hi - lo, 0.9), h, boxstyle="round,pad=0,rounding_size=0.12",
                                    mutation_aspect=1 / 18, fc=col, ec="none", zorder=3))
        gain = r.cv_to - r.cv_from
        txt = f"{gain:+.2f}".replace("-", "−")
        ax.text(max(hi, lo + 0.9) + 6, y, txt, va="center", ha="left", fontsize=9.5, color=INK2)
        # hairline connector from the end of the previous bar down to this bar's start
        ax.plot([r.cv_from, r.cv_from], [ys[i] - 0.05, y + h / 2], color=AXIS, lw=0.9, zorder=2)
    # last connector to the final dot
    ax.plot([final, final], [ys[-2] - 0.28, ys[-1] + 0.1], color=AXIS, lw=0.9, zorder=2)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("Cross-validated RMSE of the taxi-out prediction (seconds); lower is better", fontsize=9.5, labelpad=8)
    ax.set_xticks(range(300, 701, 100))
    header(fig, "Cutting the error by 58 %, one mechanism at a time",
           "RMSE on January and July 2025 (344,419 departures), each stage measured on the same folds. Bar colour = what kind of change.")
    leg = ax.legend(handles=[Patch(fc=c, label=k) for k, c in FAMILY_COLOR.items()], loc="lower right",
                    frameon=False, fontsize=9.5, handlelength=1.1, handleheight=1.1, bbox_to_anchor=(1.0, 0.04))
    for t in leg.get_texts():
        t.set_color(INK2)
    footer(fig, "Source: writeup/data/progress.csv. The last eight rows are the final chain's changes on top of the reference chain "
                "(context features, month-complete stackers, plan-row step removed, stream-transformer inputs, neighbour-label head, "
                "monthly runway calibration removed, four post-steps, second ADS-B feed removed).")
    save(fig, "fig_progress")


# ---------------------------------------------------------------- 2. two processes in one label
def fig_regimes():
    d = pd.read_csv(DATA / "label_regimes.csv")
    fig, ax = plt.subplots(figsize=(10, 4.9))
    fig.subplots_adjust(left=0.25, right=0.95, top=0.77, bottom=0.17)
    n = len(d)
    ys = np.arange(n)[::-1].astype(float)
    h = 0.34
    gap = 0.03
    for i, r in d.iterrows():
        y = ys[i]
        for off, key, col in ((+h / 2 + gap / 2, "share_rows", BLUE), (-h / 2 - gap / 2, "share_sse", ORANGE)):
            v = r[key] * 100
            ax.add_patch(FancyBboxPatch((0, y + off - h / 2), max(v, 0.25), h, boxstyle="round,pad=0,rounding_size=0.1",
                                        mutation_aspect=1 / 3.2, fc=col, ec="none", zorder=3))
            lab = f"{v:.1f} %" if v >= 0.95 else (f"{v:.2f} %" if v >= 0.01 else f"{v:.3f} %")
            ax.text(max(v, 0.25) + 0.9, y + off, lab, va="center", ha="left", fontsize=10, color=INK2)
    ax.set_xlim(0, 108)
    ax.set_ylim(-0.6, n - 0.4)
    style_axis(ax, "x")
    ax.set_yticks(ys)
    ax.set_yticklabels(d.regime, fontsize=10.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("Share of all departures / share of the total squared error", fontsize=9.5, labelpad=8)
    header(fig, "52 rows out of 344,419 carry 40 % of the squared error",
           "Reference chain (CV 285.44), January and July 2025 folds. The label is the time between a recorded off-block and take-off; "
           "some recorded off-blocks are not events.")
    leg = ax.legend(handles=[Patch(fc=BLUE, label="share of departures"), Patch(fc=ORANGE, label="share of squared error")],
                    loc="lower right", frameon=False, fontsize=10, handlelength=1.1, handleheight=1.1)
    for t in leg.get_texts():
        t.set_color(INK2)
    footer(fig, "Source: writeup/data/label_regimes.csv. 'Very long' and '1-2 h' rows are not schedule copies. "
                "A schedule copy has BLOCK = SCHED, so the label is take-off minus schedule.")
    save(fig, "fig_regimes")


# ---------------------------------------------------------------- 3. error by month, by source
def fig_months():
    d = pd.read_csv(DATA / "month_error.csv")
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    parts = [("normal", "Ordinary taxi rows", BLUE), ("copy", "Schedule copies", ORANGE),
             ("1-2h", "Long taxi, 1-2 h", YELLOW), ("gt2h", "Very long, over 2 h", INK2)]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.14)
    x = np.arange(12)
    bottom = np.zeros(12)
    gap = 1800  # ~2 px of surface between segments, in s^2 units on a 0-160k axis
    for key, label, col in parts:
        v = d[key].to_numpy(dtype=float) / 1000
        seg = np.maximum(v - gap / 1000, 0)
        ax.bar(x, seg, bottom=bottom, width=0.62, color=col, zorder=3, linewidth=0)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean squared error contribution (thousand s²)", fontsize=9.5)
    style_axis(ax, "y")
    ax.set_xlim(-0.6, 11.6)
    for xi, (tot, r) in enumerate(zip(d.total_mse / 1000, d.rmse)):
        ax.text(xi, tot + 3, f"{r:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.text(-0.55, 0.985, "RMSE (s) above each bar", transform=ax.get_xaxis_transform(), fontsize=8.5, color=MUTED,
            va="top", ha="left")
    header(fig, "Where the error comes from changes with the season",
           "Leave-one-month-out error of the base mixture on all twelve months of 2025. Missed schedule copies drive the summer rise "
           "(June to September); very long records are heavy but erratic (January as large as July).")
    leg = ax.legend(handles=[Patch(fc=c, label=l) for _, l, c in parts], loc="upper right", frameon=False, fontsize=9.5,
                    handlelength=1.1, handleheight=1.1, bbox_to_anchor=(1.0, 0.93))
    for t in leg.get_texts():
        t.set_color(INK2)
    footer(fig, "Source: writeup/data/month_error.csv (base mixture without the stackers and calibration steps, so totals are higher "
                "than the final chain's).")
    save(fig, "fig_months")


# ---------------------------------------------------------------- 4. value of the tail steps with intervals
def fig_tail():
    d = pd.read_csv(DATA / "tail_steps.csv").iloc[1:].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.9))
    fig.subplots_adjust(left=0.33, right=0.96, top=0.77, bottom=0.17)
    n = len(d)
    ys = np.arange(n)[::-1]
    ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
    dropped = d.step.str.contains("plan-row")
    for i, r in d.iterrows():
        y = ys[i]
        col = MUTED if dropped[i] else BLUE
        ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=2, solid_capstyle="round", zorder=3)
        ax.plot([r.delta], [y], "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=4)
        ax.text(1.0 if r.ci_high < 0.6 else r.ci_high + 0.1, y, f"{r.delta:+.2f}".replace("-", "−"), va="center", ha="left",
                fontsize=9.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([s + ("\n(not in the final model)" if dr else "") for s, dr in zip(d.step, dropped)], fontsize=10)
    for t, dr in zip(ax.get_yticklabels(), dropped):
        if dr:
            t.set_color(MUTED)
    ax.set_xlim(-5.6, 2.0)
    ax.set_ylim(-0.6, n - 0.4)
    style_axis(ax, "x")
    ax.set_xlabel("Change in cross-validated RMSE when the step is added (seconds); left of zero is better", fontsize=9.5, labelpad=8)
    header(fig, "Each step of the chain, against the step before it",
           "Dot = change in RMSE on the January and July 2025 folds; line = 95 % paired day-block bootstrap interval.")
    footer(fig, "Source: writeup/data/tail_steps.csv. Reference chain: the steps take it from 292.33 (rules chain) to 285.44. Grey: measured "
                "here, left out of the final model because it does not hold on held-out months.")
    save(fig, "fig_tail")


# ---------------------------------------------------------------- 5. the pipeline
def fig_pipeline():
    from matplotlib.patches import FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")
    W, H = 20, 12

    def box(x, y, title, sub, family, w=W, h=H):
        col = FAMILY_COLOR[family]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=col + "22", ec=col, lw=1.6, zorder=3))
        ax.text(x + w / 2, y + h * 0.78, title, ha="center", va="center", fontsize=10, fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2, y + h * 0.36, textwrap.fill(sub, 31), ha="center", va="center", fontsize=8, color=INK2, zorder=4, linespacing=1.3)
        return (x, y, w, h)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.3, zorder=2, shrinkA=0, shrinkB=0))

    # row 1 (top)
    a = box(1, 37, "Inputs", "challenge files, METAR, OSM, ADS-B, EUROCONTROL aggregates", "Open data")
    b = box(26, 37, "Features", "label-free: clocks, load, weather, stand history, ADS-B, record ordering", "Models and features")
    c1 = box(51, 42, "Copy classifier", "P(off-block is the schedule)", "Label structure", h=9)
    c2 = box(51, 30, "11 regressors", "physical taxi time, blended by NNLS", "Models and features", h=9)
    d = box(76, 37, "Mixture", "p \u00b7 (MVT \u2212 SCHED) + (1 \u2212 p) \u00b7 taxi", "Label structure")
    arrow((21, 43), (26, 43)); arrow((46, 43), (51, 46.5)); arrow((46, 43), (51, 34.5)); arrow((71, 46.5), (76, 45)); arrow((71, 34.5), (76, 41))
    # row 2 (bottom)
    e = box(76, 12, "Rules chain", "delayed-row gate, plan cliff, LIRF regimes, clock rules, plan-window clip", "Label structure", h=14)
    f = box(51, 12, "Stackers", "month-complete: ADS-B tracks, member disagreement, departure stream", "Models and features", h=14)
    g = box(26, 12, "Record tail", "LIRF copy classifier, near-schedule rule, daily delay calibration, duplicated records",
            "Open data", h=14)
    h = box(1, 12, "Bundle and post-steps", "stream-transformer inputs, four post-steps, then the plan-window clip", "Label structure", h=14)
    arrow((86, 37), (86, 26.2))
    arrow((76, 19), (71, 19)); arrow((51, 19), (46, 19)); arrow((26, 19), (21, 19))
    ax.text(11, 7, "final prediction", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    arrow((11, 12), (11, 9.6))
    header(fig, "The model in one picture", "Left to right along the top, then right to left along the bottom. Every box is described in docs/model.md.")
    leg = ax.legend(handles=[Patch(fc=c, label=k) for k, c in FAMILY_COLOR.items() if k != "Removed"], loc="lower right",
                    frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(1.0, -0.06), handlelength=1.1, handleheight=1.1)
    for t in leg.get_texts():
        t.set_color(INK2)
    footer(fig, "Colour = what kind of component: label structure (what the recorded time means), models and features, open data. "
                "The record tail mixes record rules with the calibration to EUROCONTROL's published daily delay (open data).")
    save(fig, "fig_pipeline")


# ---------------------------------------------------------------- 6. concentration of the squared error
def fig_concentration():
    d = pd.read_csv(DATA / "sse_curve.csv")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.77, bottom=0.17)
    ax.plot(d.rows, d.share_sse * 100, color=BLUE, lw=2.2, zorder=3)
    eq = np.logspace(0, np.log10(344419), 200)
    ax.plot(eq, eq / 344419 * 100, color=MUTED, lw=1.4, ls="--", zorder=2)
    ax.text(2.5e4, 13, "equal shares\n(every departure the same error)", fontsize=9, color=MUTED, ha="right", va="bottom")
    marks = [(2, "2 rows", (1.4, -3)), (10, "10 rows", (1.4, -3)), (100, "100 rows", (1.4, -3)), (1000, "1,000 rows", (1.4, -3))]
    for k, lab, off in marks:
        v = float(np.interp(k, d.rows, d.share_sse * 100))
        ax.plot([k], [v], "o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=4)
        ax.text(k * 1.25, v - 4.5, f"{lab}: {v:.1f} %", fontsize=10, color=INK2, va="top", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(0.9, 4e5)
    ax.set_ylim(0, 102)
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["1", "10", "100", "1,000", "10,000", "100,000"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("Number of departures, largest squared error first (log scale; 344,419 in total)", fontsize=9.5, labelpad=8)
    ax.set_ylabel("Cumulative share of the squared error", fontsize=9.5)
    style_axis(ax, "y")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    header(fig, "Two departures carry 36.5 % of the squared error",
           "Departures sorted by squared error, largest first, against the share of the total they hold. Dashed: the curve if every "
           "departure had the same error.")
    footer(fig, "Source: writeup/data/sse_curve.csv. Out-of-fold predictions of the reference chain (CV 285.44) on the January and "
                "July 2025 folds. Leaving out the largest 2 gives RMSE 227.44 instead of 285.44.")
    save(fig, "fig_concentration")


# ---------------------------------------------------------------- 7. one departure, two recordings
def fig_timeline():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.79, bottom=0.12, hspace=0.9)
    cases = [
        ("An ordinary departure", [("SCHED", 0, INK2), ("BLOCK, recorded off-block", 14, BLUE), ("AOBT_3, NM's own estimate", 16, AQUA),
                                   ("MVT, take-off", 30, ORANGE)], (14, 30), -3, 36, "label = 16 min, a physical taxi"),
        ("A schedule copy", [("SCHED = BLOCK (copied)", 0, BLUE), ("AOBT_3", 165, AQUA), ("MVT, take-off", 182, ORANGE)],
         (0, 182), -12, 215, "label = 182 min, but the aircraft taxied for about 17 min"),
    ]
    for ax, (title, pts, (a, b), lo, hi, note) in zip(axes, cases):
        ax.axhline(0, color=AXIS, lw=1.5, zorder=1)
        ax.plot([a, b], [0, 0], color=BLUE, lw=6, alpha=0.35, solid_capstyle="butt", zorder=2)
        for j, (lab, t, col) in enumerate(pts):
            up = 1 if j % 2 == 0 else -1
            ax.plot([t, t], [0, 0.55 * up], color=col, lw=1.6, zorder=3)
            ax.plot([t], [0], "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=4)
            ax.text(t, 0.7 * up, f"{lab}\n{t} min", ha="center", va="bottom" if up > 0 else "top", fontsize=9.5, color=INK2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1.8, 1.8)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=2)
        ax.set_title(note, loc="right", fontsize=10, color=INK, fontweight="bold", pad=2)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([])
    header(fig, "A recorded off-block is not always an event",
           "The label is take-off minus the recorded off-block. Illustrative times in minutes after the scheduled off-block; shaded: the span the label measures.")
    footer(fig, "Schematic, not a real flight. A flight the plan re-filed more than an hour after schedule never copies; "
                "a copy happens when the handler leaves the scheduled time in the record.")
    save(fig, "fig_timeline")


# ---------------------------------------------------------------- 8. ADS-B coverage heatmap
def fig_coverage():
    d = pd.read_csv(DATA / "adsb_coverage.csv").set_index("apt")
    cols = list(d.columns)
    mon = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep",
           "10": "Oct", "11": "Nov", "12": "Dec"}
    pos, x = [], 0.0
    for c in cols:
        if c == "2026-01":
            x += 0.6
        pos.append(x)
        x += 1
    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.12)
    r, g, b = [int(BLUE[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    for yi, apt in enumerate(d.index):
        for xi, c in zip(pos, cols):
            v = float(d.loc[apt, c])
            col = tuple(1 - (1 - ch) * (0.08 + 0.92 * v) for ch in (r, g, b))
            ax.add_patch(plt.Rectangle((xi + 0.04, yi + 0.04), 0.92, 0.92, fc=col, ec="none", zorder=2))
            ax.text(xi + 0.5, yi + 0.5, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 else INK2, zorder=3)
    ax.set_xlim(-0.1, x + 0.1)
    ax.set_ylim(len(d), 0)
    ax.set_xticks([p + 0.5 for p in pos])
    ax.set_xticklabels([mon[c[-2:]] for c in cols], fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(len(d))])
    ax.set_yticklabels(d.index, fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    y25, y26 = pos[0], pos[cols.index("2026-01")]
    ax.text(pos[5] + 0.5, len(d) + 0.55, "2025", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.text((pos[-1] + y26) / 2 + 0.5, len(d) + 0.55, "2026 (test months)", ha="center", fontsize=10, color=INK, fontweight="bold")
    header(fig, "Open ADS-B sees some airports all year and others never",
           "Share of departures with an open ADS-B ground track (take-off within 300 s), by airport and month, in per cent. "
           "Istanbul and Paris are almost invisible; Madrid and Heathrow switch on and off between years.")
    footer(fig, "Source: writeup/data/adsb_coverage.csv, adsb.lol. Darker = more of the airport's departures matched.")
    save(fig, "fig_coverage")


# ---------------------------------------------------------------- 9. error by airport
def fig_airports():
    d = pd.read_csv(DATA / "airport_error.csv")
    order = d.groupby("apt").share_sse.sum().sort_values(ascending=False).index.tolist()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True, gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.14, wspace=0.08)
    ys = np.arange(len(order))[::-1]
    for y, a in zip(ys, order):
        v = d[d.apt == a].share_sse.sum() * 100
        a1.add_patch(FancyBboxPatch((0, y - 0.3), max(v, 0.2), 0.6, boxstyle="round,pad=0,rounding_size=0.08",
                                    mutation_aspect=1 / 60, fc=BLUE, ec="none", zorder=3))
        a1.text(v + 1, y, f"{v:.1f} %", va="center", fontsize=9.5, color=INK2)
        for fold, col in (("jan", BLUE), ("jul", ORANGE)):
            r = float(d[(d.apt == a) & (d.fold == fold)].rmse_le2h.iloc[0])
            a2.plot([r], [y + (0.13 if fold == "jan" else -0.13)], "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=4)
    a1.set_xlim(0, 56)
    a2.set_xlim(40, 540)
    for ax in (a1, a2):
        ax.set_ylim(-0.6, len(order) - 0.4)
        style_axis(ax, "x")
    a1.set_yticks(ys)
    a1.set_yticklabels(order, fontsize=10.5)
    a1.set_xticks([0, 20, 40])
    a1.set_xticklabels(["0", "20 %", "40 %"])
    a1.set_xlabel("Share of the total squared error", fontsize=9.5, labelpad=8)
    a2.set_xlabel("RMSE on rows with label up to 2 h (s)", fontsize=9.5, labelpad=8)
    leg = a2.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=BLUE, label="January 2025 fold"),
                             Line2D([], [], marker="o", ls="", ms=9, color=ORANGE, label="July 2025 fold")],
                    loc="lower right", frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    header(fig, "Paris and Rome hold most of the squared error; the typical taxi is far easier",
           "Reference chain (CV 285.44) on the January and July 2025 folds. Left: share of the squared error. Right: RMSE once the very long labels are set aside.")
    footer(fig, "Source: writeup/data/airport_error.csv. LFPG's share is almost entirely its two outlier January records.")
    save(fig, "fig_airports")


# ---------------------------------------------------------------- 10. the mixture, as a picture
def fig_mixture():
    A, B, p = 180.0, 15.0, 0.3
    x = np.linspace(-20, 210, 400)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.77, bottom=0.17)
    sc = 1 / 1000
    ax.plot(x, (x - A) ** 2 * sc, color=MUTED, lw=1.6, ls="--", zorder=2)
    ax.plot(x, (x - B) ** 2 * sc, color=MUTED, lw=1.6, ls=":", zorder=2)
    ex = (p * (x - A) ** 2 + (1 - p) * (x - B) ** 2) * sc
    ax.plot(x, ex, color=BLUE, lw=2.6, zorder=3)
    m = p * A + (1 - p) * B
    ax.plot([m], [(p * (m - A) ** 2 + (1 - p) * (m - B) ** 2) * sc], "o", ms=11, color=BLUE, mec=SURFACE, mew=2, zorder=5)
    ax.plot([A], [0], "o", ms=9, color=ORANGE, mec=SURFACE, mew=2, zorder=5)
    ax.plot([B], [0], "o", ms=9, color=AQUA, mec=SURFACE, mew=2, zorder=5)
    ax.text(A, 2.2, "answer if copied\n(180 min)", ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.text(B, 2.2, "answer if physical\n(15 min)", ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.annotate(f"best single prediction: {m:.1f} min\n= {p:.1f} · 180 + {1 - p:.1f} · 15", xy=(m, 6.4), xytext=(m + 14, 21.5), fontsize=10,
                color=INK, fontweight="bold", va="bottom", ha="center", zorder=7,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1, shrinkA=3, shrinkB=2))
    ax.text(166, 27, "cost if it was a physical taxi (dotted)", fontsize=9, color=INK2, ha="right")
    ax.text(14, 31.5, "cost if it was a copy (dashed)", fontsize=9, color=INK2, ha="left")
    ax.text(-15, 14.8, "expected cost\nat copy probability 0.3", fontsize=9.5, color=INK2, ha="left", va="bottom")
    ax.set_xlim(-20, 210)
    ax.set_ylim(0, 36)
    ax.set_xlabel("Predicted label (minutes)", fontsize=9.5, labelpad=8)
    ax.set_ylabel("Squared error (thousand min²)", fontsize=9.5)
    style_axis(ax, "y")
    header(fig, "Under squared error the best prediction is the probability-weighted mix",
           "One departure with a 30 % chance that its recorded off-block is the schedule. Neither answer is the best bet; the mix is.")
    footer(fig, "Schematic with illustrative numbers. The model's copy probability is calibrated, which matters more than its ranking.")
    save(fig, "fig_mixture")


# ---------------------------------------------------------------- 11. calibration
def fig_calibration():
    d = pd.read_csv(DATA / "calibration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.2, wspace=0.18)
    for ax, view, ttl in zip(axes, ("by prediction", "by true value"), ("Grouped by what the model predicted", "Grouped by what actually happened")):
        w = d[d.view == view].reset_index(drop=True)
        x = np.arange(len(w))
        ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
        ax.bar(x, w.bias, width=0.62, color=BLUE, zorder=3)
        log = view == "by true value"
        if log:
            ax.set_yscale("symlog", linthresh=50)
            ax.set_yticks([-5000, -1000, -250, -50, 0, 50, 250])
            ax.set_yticklabels(["−5,000", "−1,000", "−250", "−50", "0", "+50", "+250"], fontsize=8.5)
            ax.set_ylim(-20000, 1100)
        else:
            ax.set_ylim(-60, 125)
            ax.tick_params(axis="y", labelsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(w.band.str.replace("0 or less", "≤ 0"), rotation=35, ha="right", fontsize=8.5)
        ax.set_title(ttl, loc="left", fontsize=10.5, color=INK, pad=6)
        style_axis(ax, "y")
        for xi, b, n in zip(x, w.bias, w.n):
            off = (0.15 * abs(b) + 6) if log else 4
            ax.text(xi, b + off * (1 if b > 0 else -1), f"{b:+.0f}".replace("-", "−"), ha="center",
                    va="bottom" if b > 0 else "top", fontsize=8, color=INK2)
    axes[0].set_ylabel("Mean error, prediction − truth (s)", fontsize=9)
    axes[1].set_ylabel("Mean error (s), axis linear within ±50 s, log beyond", fontsize=9)
    header(fig, "The model is right on average where the taxi is predictable, and shrinks the extremes",
           "Mean error by band of the prediction (left, all 344,419 departures) and by band of the true label (right, the 326,969 "
           "that are not schedule copies). "
           "Regression to the mean at the long end is what squared error asks for.")
    footer(fig, "Source: writeup/data/calibration.csv. Reference chain, January and July 2025 folds. Bars with few rows (48, 79, 102, 52) are noisy.")
    save(fig, "fig_calibration")


# ---------------------------------------------------------------- 12. rehearsal by airport
def fig_rehearsal():
    d = pd.read_csv(DATA / "rehearsal.csv")
    d = d[(d.apt != "all") & (d.month != "all")]
    apts = sorted(d.apt.unique(), key=lambda a: -d[d.apt == a].rmse_le2h.mean())
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.14)
    x = np.arange(len(apts))
    w = 0.34
    for off, mon, col in ((-w / 2 - 0.02, "2025-02", BLUE), (w / 2 + 0.02, "2025-06", ORANGE)):
        v = [float(d[(d.apt == a) & (d.month == mon)].rmse_le2h.iloc[0]) for a in apts]
        ax.bar(x + off, v, width=w, color=col, zorder=3)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 6, f"{vi:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(apts, fontsize=10)
    ax.set_ylabel("RMSE on rows with label up to 2 h (s)", fontsize=9.5)
    style_axis(ax, "y")
    leg = ax.legend(handles=[Patch(fc=BLUE, label="February 2025 (held out)"), Patch(fc=ORANGE, label="June 2025 (held out)")],
                    loc="upper right", frameon=False, fontsize=9.5, handlelength=1.1, handleheight=1.1)
    for t in leg.get_texts():
        t.set_color(INK2)
    header(fig, "Rehearsal on two held-out months of 2025",
           "Reference chain, trained on the other ten months of 2025 and scored on February and June. All-row RMSE is 208 and 274 s "
           "(airport-mean baseline 606 and 519 s); this shows the rows up to 2 h so the airports compare.")
    footer(fig, "Source: writeup/data/rehearsal.csv. LTFM February is the snow month; LIRF June's all-row RMSE (805 s) is driven by long no-plan rows.")
    save(fig, "fig_rehearsal")


# ---------------------------------------------------------------- 13. reliability of the copy probability
def fig_copy_reliability():
    d = pd.read_csv(DATA / "copy_calibration.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.15, wspace=0.28)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, ls="--", zorder=1)
    for grp, col, mk, lab in (("all", BLUE, "o", "all departures"), ("LIRF", ORANGE, "s", "Rome (LIRF) only")):
        w = d[d.group == grp]
        ax.plot(w.mean_p, w.copy_rate, color=col, lw=1.4, marker=mk, ms=7, mec=SURFACE, mew=1.5, zorder=3, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t * 100:.0f} %" for t in ticks])
    ax.set_yticklabels([f"{t * 100:.0f} %" for t in ticks])
    ax.set_xlabel("Predicted copy probability p, mean of the bin", fontsize=9.5, labelpad=8)
    ax.set_ylabel("Observed share of copies in the bin", fontsize=9.5)
    style_axis(ax, "both")
    ax.text(0.62, 0.55, "perfect calibration", rotation=40, fontsize=9, color=MUTED, ha="center", va="center")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    w = d[d.group == "all"].reset_index(drop=True)
    y = np.arange(len(w))[::-1]
    ax2.barh(y, w.n, color=BLUE, height=0.62, zorder=3)
    for yi, n, mp, cr in zip(y, w.n, w.mean_p, w.copy_rate):
        ax2.text(n * 1.15, yi, f"{n:,}   {mp * 100:.1f} → {cr * 100:.1f} %", va="center", fontsize=8, color=INK2)
    ax2.set_xscale("log")
    ax2.set_xlim(200, 2e7)
    ax2.set_yticks(y)
    ax2.set_yticklabels([b.replace("-", "–") for b in w.bin], fontsize=8.5)
    ax2.set_xlabel("Departures in the bin (log scale)", fontsize=9.5, labelpad=8)
    ax2.set_title("Bins of p: rows, predicted → observed", loc="left", fontsize=10, color=INK, pad=6)
    style_axis(ax2, "x")
    header(fig, "The copy probability is close to calibrated",
           "Out-of-fold copy probability on the January and July 2025 folds, in ten bins, against the share of copies "
           "(|BLOCK − SCHED| ≤ 30 s). Slightly low above 10 % (58 → 61 %, 79 → 83 %), slightly high in the 0–1 % bin.")
    footer(fig, "Source: writeup/data/copy_calibration.csv. AUC 0.870; log loss 0.148 against 0.200 for a constant at the 5.1 % base rate; "
                "Brier 0.041 against 0.048. Rome: AUC 0.866, log loss 0.309 against 0.452.")
    save(fig, "fig_copy_reliability")


# ---------------------------------------------------------------- 14. where the remaining error is
def fig_error_budget():
    b = pd.read_csv(DATA / "error_budget.csv")
    c = pd.read_csv(DATA / "error_bulk_cells.csv").head(10)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.75})
    fig.subplots_adjust(left=0.2, right=0.97, top=0.76, bottom=0.2)
    col = {"datestamp": INK2, "giant": MUTED, "far": "#bdbcb4", "monster": YELLOW, "bulk": BLUE}
    ys = np.arange(len(b))[::-1]
    a1.barh(ys, b.share_pct, color=[col[k] for k in b.key], height=0.62, zorder=3)
    for y, r in zip(ys, b.itertuples()):
        a1.text(r.share_pct + 1, y, f"{r.share_pct:.1f} %  ({r.rows:,} rows)", va="center", fontsize=8.5, color=INK2)
    a1.set_yticks(ys)
    a1.set_yticklabels([x.replace(" (", "\n(") for x in b.label], fontsize=8.5)
    a1.set_xlim(0, 75)
    a1.set_xlabel("Share of the total squared error", fontsize=9.5, labelpad=8)
    a1.set_title("Five kinds of row", loc="left", fontsize=10, color=INK, pad=6)
    style_axis(a1, "x")
    hubs = {"LTFM", "LFPG", "EGLL", "LIRF"}
    ys2 = np.arange(len(c))[::-1]
    hub = [(r.apt in hubs and r.adsb == "no_adsb") for r in c.itertuples()]
    a2.barh(ys2, c.share_pct, color=[ORANGE if h else BLUE for h in hub], height=0.62, zorder=3)
    for y, r in zip(ys2, c.itertuples()):
        a2.text(r.share_pct + 0.2, y, f"{r.share_pct:.1f} %", va="center", fontsize=8.5, color=INK2)
    a2.set_yticks(ys2)
    a2.set_yticklabels([f"{r.apt}, {'no track' if r.adsb == 'no_adsb' else 'with track'}" for r in c.itertuples()], fontsize=8.5)
    a2.set_xlim(0, 12)
    a2.set_xlabel("Share of the total squared error", fontsize=9.5, labelpad=8)
    a2.set_title("The ordinary rows, by airport and track", loc="left", fontsize=10, color=INK, pad=6)
    style_axis(a2, "x")
    hs = c[hub].share_pct.sum()
    header(fig, "Half the error is unreachable; the rest sits at hubs without tracks",
           "Shipped chain on four held-out months of 2025 (671,292 departures). Left: squared error by kind of row. "
           "Right: the ordinary rows by airport and ground track; orange = the four hubs without tracks.")
    footer(fig, "Source: writeup/data/error_budget.csv, error_bulk_cells.csv. Artefacts = date stamps, giant labels, far schedule copies; "
                f"the four orange bars hold {hs:.1f} % of all squared error.")
    save(fig, "fig_error_budget")


# ---------------------------------------------------------------- 15. predicted against recorded
def _density_panel(ax, cmap_name="Blues"):
    """Heat map of the final chain's out-of-fold predictions against the recorded labels (counts on a 30 s grid, log colour)."""
    from matplotlib.colors import LogNorm
    g = pd.read_csv(DATA / "pred_actual.csv")
    n = int(2400 / 30)
    m = np.zeros((n, n))
    m[(g.p_lo // 30).to_numpy(), (g.y_lo // 30).to_numpy()] = g.n.to_numpy()
    m[m == 0] = np.nan
    im = ax.imshow(m, origin="lower", extent=(0, 2400, 0, 2400), aspect="equal", cmap=cmap_name,
                   norm=LogNorm(vmin=1, vmax=np.nanmax(m)), interpolation="nearest", zorder=2)
    ax.plot([0, 2400], [0, 2400], color=ORANGE, lw=1.2, zorder=3)
    ax.set_xlim(0, 2400)
    ax.set_ylim(0, 2400)
    ticks = [0, 600, 1200, 1800, 2400]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t // 60}" for t in ticks])
    ax.set_yticklabels([f"{t // 60}" for t in ticks])
    ax.set_xlabel("Recorded taxi-out time (minutes)", fontsize=9.5, labelpad=6)
    ax.set_ylabel("Predicted (minutes)", fontsize=9.5, labelpad=6)
    ax.tick_params(length=0)
    ax.set_facecolor(SURFACE)
    return im


def fig_density():
    s = pd.read_csv(DATA / "pred_actual_summary.csv").set_index("name").value
    h = pd.read_csv(DATA / "pred_error_hist.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.28})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.19)
    im = _density_panel(a1)
    cb = fig.colorbar(im, ax=a1, fraction=0.04, pad=0.02, shrink=0.8)
    cb.set_label("departures per 30 s \u00d7 30 s cell", fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    a1.set_title("Prediction against record, every departure", loc="left", fontsize=10, color=INK, pad=6)
    centre = h.err_lo + 5
    a2.bar(centre, h.n / h.n.sum() * 100, width=9, color=BLUE, zorder=3)
    a2.axvline(0, color=ORANGE, lw=1.2, zorder=4)
    a2.set_xlim(-300, 300)
    a2.set_xlabel("Prediction minus recorded label (seconds)", fontsize=9.5, labelpad=6)
    a2.set_ylabel("Share of departures per 10 s (%)", fontsize=9.5, labelpad=6)
    a2.set_title("How far off", loc="left", fontsize=10, color=INK, pad=6)
    style_axis(a2, "y")
    a2.text(0.97, 0.95, f"{s.share_within_120s * 100:.0f} % within 2 min\n{s.share_within_300s * 100:.0f} % within 5 min\n"
            f"median error {s.median_abs_error_s:.0f} s", transform=a2.transAxes, ha="right", va="top", fontsize=9.5, color=INK2, linespacing=1.5)
    header(fig, "Predictions follow the recorded taxi time, and most are within two minutes",
           "Final chain, January and July 2025 out of fold (344,419 departures). Left: darker = more departures; the orange line is a perfect prediction. "
           "Right: the error of every departure.")
    footer(fig, f"Source: writeup/data/pred_actual.csv, pred_error_hist.csv. {int(s.rows_label_at_or_above_2400s):,} departures with a label of 40 minutes or more lie "
                f"off the left chart; they hold {s.share_of_squared_error_label_at_or_above_2400s * 100:.0f} % of the squared error (docs/error-budget.md). The right chart clips errors at \u00b1600 s.")
    save(fig, "fig_density")


# ---------------------------------------------------------------- 16. the hero: the result and the evidence side by side
def fig_hero():
    d = pd.read_csv(DATA / "progress.csv")
    rows, pend = [], []

    def flush():
        if pend:
            lab = pend[0].stage if len(pend) == 1 else f"{len(pend)} smaller steps"
            rows.append((lab, pend[0].family if len(pend) == 1 else "Models and features", float(pend[0].cv_from), float(pend[-1].cv_to)))
            pend.clear()
    for r in d.itertuples():
        if r.cv_from - r.cv_to >= 5:
            flush()
            rows.append((r.stage, r.family, float(r.cv_from), float(r.cv_to)))
        else:
            pend.append(r)
    flush()
    short = {"Predict the residual on the NM off-block clock": "Predict against the clock\nthat is recorded",
             "Mixture: probability that BLOCK is a copy of SCHED": "Probability that the record\nis a copy of the schedule",
             "LIRF record-regime post-processing": "Rome record regimes",
             "Target clip, deeper trees, stand and runway history": "Deeper trees, stand and\nrunway history",
             "CatBoost and denoising-autoencoder members + blend": "More members and a blend",
             "Clock rules: date shift, gate hold, LIRF copy": "Clock rules",
             "ADS-B ground-track residual stacker": "ADS-B ground tracks",
             "MLP and LOBT members, second stacker, LIRF rules": "Neural members, 2nd stacker"}
    fig = plt.figure(figsize=(11.5, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], left=0.17, right=0.95, top=0.78, bottom=0.14, wspace=0.42)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    n = len(rows) + 2
    ys = np.arange(n)[::-1]
    base, final = float(d.cv_from.iloc[0]), float(d.cv_to.iloc[-1])
    for y, v in ((ys[0], base), (ys[-1], final)):
        a1.plot([v], [y], "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=5)
        a1.text(v + 10, y, f"{v:.0f} s", va="center", ha="left", fontsize=10, fontweight="bold", color=INK)
    for i, (st, fam, a, b) in enumerate(rows):
        y = ys[i + 1]
        a1.add_patch(FancyBboxPatch((b, y - 0.3), max(a - b, 1.0), 0.6, boxstyle="round,pad=0,rounding_size=0.1", mutation_aspect=1 / 18,
                                    fc=FAMILY_COLOR[fam], ec="none", zorder=3))
        a1.text(a + 10, y, f"\u2212{a - b:.0f}", va="center", ha="left", fontsize=9.5, color=INK2)
        a1.plot([a, a], [ys[i] - 0.05, y + 0.3], color=AXIS, lw=0.9, zorder=2)
    a1.plot([final, final], [ys[-2] - 0.3, ys[-1] + 0.1], color=AXIS, lw=0.9, zorder=2)
    a1.set_yticks(ys)
    a1.set_yticklabels(["Airport-mean baseline"] + [short.get(r[0], r[0]) for r in rows] + ["Final model"], fontsize=8.8, linespacing=1.15)
    a1.get_yticklabels()[0].set_fontweight("bold")
    a1.get_yticklabels()[-1].set_fontweight("bold")
    a1.set_xlim(240, 740)
    a1.set_ylim(-0.7, n - 0.3)
    a1.set_xticks([300, 400, 500, 600, 700])
    a1.set_xlabel("Cross-validated RMSE (seconds), lower is better", fontsize=9.5, labelpad=6)
    style_axis(a1, "x")
    a1.set_title(f"{base:.0f} s \u2192 {final:.0f} s, stage by stage", loc="left", fontsize=10, color=INK, pad=6)
    im = _density_panel(a2)
    a2.set_title("Predicted against recorded, 344,419 departures", loc="left", fontsize=10, color=INK, pad=6)
    cb = fig.colorbar(im, ax=a2, fraction=0.045, pad=0.02, shrink=0.8)
    cb.set_label("departures per cell", fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    a2.text(0.04, 0.96, "orange: perfect", transform=a2.transAxes, fontsize=8.5, color=ORANGE, va="top")
    leg = a1.legend(handles=[Patch(fc=c, label=k) for k, c in FAMILY_COLOR.items() if k != "Removed"], loc="lower right", frameon=False,
                    fontsize=8.8, handlelength=1.1, handleheight=1.1)
    for t in leg.get_texts():
        t.set_color(INK2)
    header(fig, f"Taxi-out time at ten European airports: {base:.0f} s \u2192 {final:.0f} s RMSE",
           "Most of the gain comes from understanding the label, not from a better regressor. January and July 2025, out of fold.", y=0.97)
    footer(fig, "Source: writeup/data/progress.csv, pred_actual.csv. Neighbouring steps of under 5 s are summed in one bar; every step is in fig_progress.")
    save(fig, "fig_hero")


def fig_taxi_schematic():
    """Conceptual schematic. The picture is an AI-generated illustration (illustrations/PROMPTS.md); every label and lane is drawn here."""
    img = plt.imread(HERE / "illustrations" / "taxi_schematic_base.jpg")
    h, w = img.shape[:2]
    k = w / 1876.0  # label positions below are in the 1876 px frame of the original
    fig = plt.figure(figsize=(11.0, 8.0))
    top = fig.add_axes([0.02, 0.405, 0.96, 0.46])
    top.imshow(img, extent=(0, 1876, 838, 0), aspect="auto")
    top.set_xlim(0, 1876)
    top.set_ylim(838, 0)
    top.axis("off")
    badges = [(1, 235, 560), (2, 352, 612), (3, 640, 560), (4, 1250, 205), (5, 1612, 520)]
    for n, x, y in badges:
        top.text(x, y, str(n), ha="center", va="center", fontsize=10, fontweight="bold", color=SURFACE,
                 bbox=dict(boxstyle="circle,pad=0.28", fc=INK, ec=SURFACE, lw=1.5), zorder=5)
    lanes = fig.add_axes([0.02, 0.085, 0.96, 0.285])
    lanes.set_xlim(0, 1876)
    lanes.set_ylim(0, 5.7)
    lanes.axis("off")
    lanes.text(938, 5.35, "\u2003\u2003".join(["1\u00a0Stand", "2\u00a0Push-back", "3\u00a0Taxi", "4\u00a0Queue at the holding point", "5\u00a0Take-off"]), ha="center", va="center",
               fontsize=9.5, fontweight="bold", color=INK)
    lanes.plot([0, 1876], [4.95, 4.95], color=AXIS, lw=0.8)
    rows = [(4.2, "Flight plan and schedule", "Network Manager, airport schedule", MUTED),
            (3.2, "Recorded off-block", "challenge file: the label starts here", BLUE),
            (2.2, "ADS-B ground track", "open data: where the aircraft actually is", AQUA),
            (1.2, "Recorded take-off", "challenge file: the label ends here", BLUE)]
    for y, name, sub, c in rows:
        lanes.plot([0, 1876], [y, y], color=GRID, lw=0.8, zorder=1)
        lanes.text(10, y + 0.4, name, ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")
        lanes.text(430, y + 0.4, sub, ha="left", va="center", fontsize=8.5, color=MUTED)
    x0, x1, xs = 352, 1755, 235
    lanes.plot([xs], [4.2], "D", ms=8, mfc=SURFACE, mec=MUTED, mew=1.8, zorder=4)
    lanes.text(xs + 38, 4.2, "scheduled off-block: the plan, before the day", ha="left", va="center", fontsize=8.5, color=INK2)
    lanes.plot([x0], [3.2], "o", ms=9, color=BLUE, mec=SURFACE, mew=1.5, zorder=4)
    lanes.text(x0 + 38, 3.2, "an event, or for about 1 departure in 20 the scheduled time copied into the record", ha="left", va="center", fontsize=8.5, color=INK2)
    lanes.plot([x0, 1700], [2.2, 2.2], color=AQUA, lw=15, solid_capstyle="butt", zorder=3)
    lanes.text((x0 + 1700) / 2, 2.2, "seen from push-back to the runway; not every flight has a track", ha="center", va="center", fontsize=8.5, color=INK, zorder=4)
    lanes.plot([x1], [1.2], "o", ms=9, color=BLUE, mec=SURFACE, mew=1.5, zorder=4)
    lanes.text(x1 - 38, 1.2, "the aircraft leaves the ground", ha="right", va="center", fontsize=8.5, color=INK2)
    lanes.annotate("", xy=(x1, 0.38), xytext=(x0, 0.38), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.4, shrinkA=0, shrinkB=0))
    lanes.plot([x0, x0], [0.2, 3.2], color=BLUE, lw=0.8, ls=(0, (2, 3)), zorder=2)
    lanes.plot([x1, x1], [0.2, 1.2], color=BLUE, lw=0.8, ls=(0, (2, 3)), zorder=2)
    lanes.text((x0 + x1) / 2, 0.02, "taxi-out time: recorded take-off minus recorded off-block, the number the model estimates",
               ha="center", va="top", fontsize=9, color=BLUE, fontweight="bold")
    header(fig, "One departure, from stand to take-off",
           "What the label measures, and what each data source sees. The airport picture is an illustration, not data; not to scale.", y=0.975)
    footer(fig, "Illustration generated with an image model (prompt in illustrations/PROMPTS.md); labels, lanes and markers are drawn by make_figures.py.")
    save(fig, "fig_taxi_schematic")


# ---------------------------------------------------------------- 18. what is in the ensemble
def fig_ensemble():
    from matplotlib.patches import FancyArrowPatch
    d = pd.read_csv(DATA / "ensemble_members.csv")  # numbers from docs/model.md, "The eleven regressors" table
    fam_col = {"CatBoost": BLUE, "MLP": ORANGE, "Autoencoder": AQUA}
    fig = plt.figure(figsize=(11, 11.2))
    ax = fig.add_axes([0.02, 0.055, 0.96, 0.825])
    ax.set_xlim(0, 100)
    ax.set_ylim(-7, 100)
    ax.axis("off")

    def box(x, y, w, h, title, sub, col, lw=1.6, sub_w=26, ghost=False, tsize=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=col + "22" if not ghost else "#f1f0ec",
                                    ec=col, lw=lw, ls="--" if ghost else "-", zorder=3))
        ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=MUTED if ghost else INK, zorder=4)
        ax.text(x + w / 2, y + h - 4.1, textwrap.fill(sub, sub_w), ha="center", va="top", fontsize=7.8,
                color=MUTED if ghost else INK2, zorder=4, linespacing=1.25)

    def arrow(p, q, col=MUTED):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=col, lw=1.3, zorder=2, shrinkA=0, shrinkB=0))

    # --- A. the flow
    ax.text(0, 100.5, "Where it sits in the chain", fontsize=10.5, fontweight="bold", color=INK, va="bottom")
    W = 14.6
    xs = [0, 17, 34, 51, 68, 85]
    box(xs[0], 90.5, W, 8.5, "Copy classifier", "probability p of a copy", BLUE, sub_w=24)
    box(xs[0], 79.5, W, 9.5, "11 regressors", "physical taxi, NNLS blend", ORANGE, lw=2.6, sub_w=24)
    box(xs[1], 81, W, 15, "Mixture", "p \u00b7 (take-off \u2212 schedule) + (1 \u2212 p) \u00b7 blended taxi", BLUE, sub_w=19)
    box(xs[2], 81, W, 15, "Rules chain", "delayed-row gate, plan cliff, LIRF regimes, clock rules, plan-window clip", BLUE, sub_w=19)
    box(xs[3], 81, W, 15, "Stackers", "month-complete; four models averaged, correct the chain's residual", ORANGE, lw=2.6, sub_w=19)
    box(xs[4], 81, W, 15, "Record tail", "LIRF copy classifier, near-schedule rule, daily delay calibration", AQUA, sub_w=19)
    box(xs[5], 81, W, 15, "Four post-steps", "then the plan-window clip: final prediction", BLUE, sub_w=19)
    arrow((W, 94.5), (xs[1], 91.5)); arrow((W, 84.5), (xs[1], 86))
    for a, b in zip(xs[1:5], xs[2:]):
        arrow((a + W, 88.5), (b, 88.5))

    # --- B. the eleven regressors
    ax.text(0, 76.5, "The eleven regressors, blended into the physical taxi", fontsize=10.5, fontweight="bold", color=INK, va="bottom")
    cx = {"n": 0.5, "learner": 3.2, "anchor": 28, "inputs": 34, "cv": 63.5, "bar": 70}
    ytop = 73.2
    for k, t in [("n", "#"), ("learner", "learner"), ("anchor", "anchor"), ("inputs", "inputs beyond the base set"),
                 ("cv", "CV RMSE alone"), ("bar", "blend weight")]:
        ax.text(cx[k] + (3.4 if k == "cv" else 0), ytop, t, fontsize=8.3, color=INK2, fontweight="bold", va="center",
                ha="right" if k == "cv" else "left")
    ax.plot([0, 100], [ytop - 1.4, ytop - 1.4], color=AXIS, lw=0.8)
    rh, bmax = 2.95, 0.25
    barw = 24.0
    for i, r in d.iterrows():
        y = ytop - 3.4 - i * rh
        col = fam_col[r.family]
        zero = r.blend_weight == 0
        if zero:
            ax.add_patch(plt.Rectangle((0, y - rh / 2 + 0.1), 100, rh - 0.2, fc="#f1f0ec", ec="none", zorder=0))
        tc = MUTED if zero else INK
        ax.text(cx["n"], y, str(r.member), fontsize=8.3, color=tc, va="center", fontweight="bold")
        learner = "denoising autoencoder + MLP head" if r.family == "Autoencoder" else r.learner
        ax.text(cx["learner"], y, learner, fontsize=8.3, color=tc, va="center")
        ax.text(cx["anchor"], y, r.anchor, fontsize=8.3, color=tc, va="center", family="DejaVu Sans Mono")
        ax.text(cx["inputs"], y, r.extra_inputs.replace("EUROCONTROL data", "EUROCONTROL"),
                fontsize=7.3, color=INK2 if not zero else MUTED, va="center")
        ax.text(cx["cv"] + 3.4, y, f"{r.cv_alone:.2f}", fontsize=8.3, color=INK2 if not zero else MUTED, va="center", ha="right")
        w = r.blend_weight / bmax * barw
        if zero:
            ax.add_patch(plt.Rectangle((cx["bar"], y - 0.85), 0.6, 1.7, fc=col + "55", ec=col, lw=0.8, hatch="////", zorder=3))
            ax.text(cx["bar"] + 1.6, y, "0: feeds the disagreement features only", fontsize=7.6, color=MUTED, va="center", style="italic")
        else:
            ax.add_patch(plt.Rectangle((cx["bar"], y - 0.95), w, 1.9, fc=col, ec="none", zorder=3))
            ax.text(cx["bar"] + w + 0.7, y, f"{r.blend_weight:.3f}", fontsize=8.3, color=INK, va="center", fontweight="bold")
    yend = ytop - 3.4 - 10 * rh - rh / 2
    leg = ax.legend(handles=[Patch(fc=BLUE, label="CatBoost (7)"), Patch(fc=ORANGE, label="MLP 4 \u00d7 1024 (3)"),
                             Patch(fc=AQUA, label="denoising autoencoder (1)"),
                             Patch(fc="#f1f0ec", ec=MUTED, hatch="////", label="weight 0, still feeds the stackers")],
                    loc="upper left", bbox_to_anchor=(0.0, (yend - 0.6 + 7) / 107), frameon=False, fontsize=8.3, ncol=4, handlelength=1.1,
                    handleheight=1.1, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)

    # --- C. the stream transformers
    yc = yend - 9.5
    ax.text(0, yc + 2.0, "The stream transformers that feed the stackers", fontsize=10.5, fontweight="bold", color=INK, va="bottom")
    CH = 10.2
    rows = [(yc - CH, CH), (yc - 2 * CH - 1.2, CH), (yc - 3 * CH - 2.4, CH)]
    ybot = yc - 3 * CH - 2.4
    box(0, ybot, 22, yc - ybot, "One network", "Every movement of an airport, departures and arrivals, is a token in time order. "
        "Chunks of 512 tokens, a learned time decay per attention head. Each variant has an out-of-fold side and a refit for each rehearsal and the test months.",
        MUTED, sub_w=30)
    cards = [
        ("Dropout transformer  ts_kdu3", "ADS-B modality dropout and distillation from an averaged copy of itself. Five stacker columns: prediction, gap to chain, spread, "
         "P(label > 1,800 s), P(label > 3,600 s). Replaced in the final chain by stx_nb.", MUTED, True),
        ("Context transformer  stx_ctx", "Same network with the departure-stream windows on each departure token, without the chain's own predictions. "
         "Joins the eleven regressors in the member-disagreement features.", ORANGE, False),
        ("Neighbour-label transformer  stx_nb", "The dropout transformer plus one head: the shrunken leave-one-out mean of the clipped residual of the other departures within \u00b130 min, "
         "per airport (nbm) and per airport and runway (nbr). Its five columns plus nbm30 and nbr30 go to the stackers.", ORANGE, False),
    ]
    for (y, h), (t, s, c, ghost) in zip(rows, cards):
        box(25, y, 52, h, t, s, c, sub_w=84, ghost=ghost, tsize=9)
    sx, sw = 82, 17.6
    box(sx, ybot, sw, yc - ybot, "Stackers", "Month-complete: XGBoost, CatBoost, LightGBM and XGBoost with ADS-B coverage weights, equal average. "
        "Trained on the January/July out-of-fold rows plus five rehearsal pairs.", ORANGE, lw=2.6, sub_w=24, tsize=9.5)
    for y, h in rows:
        arrow((22, y + h / 2), (25, y + h / 2))
    arrow((77, rows[1][0] + CH / 2), (sx, rows[1][0] + CH / 2 + 3), ORANGE)
    arrow((77, rows[2][0] + CH / 2), (sx, rows[2][0] + CH / 2 + 3), ORANGE)
    header(fig, "What is in the ensemble",
           "Eleven regressors, two stream-transformer members and four stackers. Every number is from docs/model.md.")
    footer(fig, "Blend weights: NNLS on pooled out-of-fold predictions of non-copy rows. CV RMSE alone: the member as the only regressor in the mixture, before any rule, "
                "so these sit above the final chain's 279.61. Thick outlines mark the two boxes detailed below. LOBT: last filed off-block time; AOBT_3: the Network Manager's off-block estimate.")
    save(fig, "fig_ensemble")


# ---------------------------------------------------------------- 19. which model does what, along the whole pipeline
def fig_models():
    """Every stage of the final chain as cards: what kind of component it is and what job it does. Text from docs/model.md."""
    from matplotlib.patches import Circle
    LEARNED, FITTED, RULE, INPUT = BLUE, AQUA, MUTED, YELLOW
    TAG = {LEARNED: "learned model", FITTED: "fitted constants, no learner", RULE: "fixed rule", INPUT: "inputs, not a model"}
    members = pd.read_csv(DATA / "ensemble_members.csv")
    wsum = members.groupby("family").blend_weight.sum()  # CatBoost 0.529, MLP 0.433, Autoencoder 0.036
    CPU = 1.42  # characters per axis unit at 7.6 pt (an axis unit is 0.1056 in in both directions)

    def lines(text, w):
        return textwrap.wrap(text, int((w - 2.0) * CPU))

    def chip_h(c):
        return 1.0 + 1.7 * (1 + (len(c["title"]) >= int(c["w"] * 1.25))) + 1.3 + len(lines(c["text"], c["w"])) * 1.3 + 1.0

    # stage = (number, title, [chips], extra height, label)
    W0, GAP, X0 = 84.0, 1.2, 16.0

    def row(*specs):
        ws = [sp[0] for sp in specs]
        tot = sum(ws)
        out = []
        for wfrac, title, kind, text in specs:
            out.append(dict(w=(W0 - GAP * (len(specs) - 1)) * wfrac / tot, title=title, col=kind, text=text))
        return out

    stages = [
        ("1", "Copy classifier", row((1, "XGBoost classifier", LEARNED,
            "Predicts p, the chance that a departure's recorded off-block copies its schedule (|BLOCK \u2212 SCHED| \u2264 30 s). Depth 8, learning rate 0.05, "
            "early-stopped on log loss. Reads challenge data only: clocks, recording process, arrival-side habits, load, stand and runway history. "
            "Out of fold its p is close to calibrated (AUC 0.870): 30 % means 30 %.")), 0),
        ("2", "Eleven regressors and their blend", row(
            (1, "CatBoost \u00d7 7", LEARNED, f"Blend weight {wsum['CatBoost']:.3f} in total. Four members carry it (LOBT or AOBT_3 anchor); three weigh 0 and only feed the stackers' member-disagreement features."),
            (1, "MLP 4 \u00d7 1024 \u00d7 3", LEARNED, f"Blend weight {wsum['MLP']:.3f} in total: members 2 and 4 carry it, member 10 weighs 0."),
            (1, "Denoising autoencoder + MLP head", LEARNED, f"Blend weight {wsum['Autoencoder']:.3f}. Swap noise 0.25, 2048 units.")), 8.0),
        ("3", "Mixture", row((1, "pred = p \u00b7 (d_sched + \u03b4) + (1 \u2212 p) \u00b7 blended regressors", RULE,
            "The prediction is the probability-weighted mix of the two branches: the copy answer d_sched + \u03b4 (take-off minus schedule) and the blended physical taxi. "
            "With p and the regressors calibrated, this is the squared-error optimum.")), 0),
        ("4", "Rules chain", row(
            (1, "Delayed-row gate: CatBoost classifier q", LEARNED,
             "The same copy flag, fitted only on departures 900\u201320,000 s late, cross-fitted by month, averaged over 5 seeds. Mixed in as p + w (q \u2212 p), w = 0.75 at LIRF and 0 elsewhere."),
            (1, "Plan cliff, LIRF regimes, clock rules, plan-window clip", FITTED,
             "Rules for known record faults, not models. The plan cliff sets p = 0 when LOBT \u2212 SCHED > 3,600 s. LIRF regimes split a no-plan departure into copy, datestamp or real taxi with rates per cell fitted on training months.")), 0),
        ("5", "Month-complete stackers", row(
            (2.0, "Four gradient-boosting stackers, equal average", LEARNED,
             "XGBoost depth 7, CatBoost, LightGBM extra trees (127 leaves), and XGBoost with rows weighted to the 2026 ADS-B coverage mix. Each predicts the chain's residual y \u2212 pred "
             "(clipped at \u00b11,800 s); it is added back. A second model of the same learner corrects rows without an ADS-B match. Trained on January/July out-of-fold rows plus five rehearsal pairs, "
             "so it sees every month of 2025."),
            (1.4, "Stacker inputs", INPUT,
             "ADS-B timings, stops and path; the chain's prediction and clock gaps; airport; pushback features; departure-stream context; member disagreement (each regressor's gap to the blend, spread, range)."),
            (1.5, "Context transformer stx_ctx", LEARNED,
             "A transformer over every movement of an airport in time order (chunks of 512, learned time decay per head), with the departure-stream windows on each departure token. Its prediction joins the eleven regressors in the disagreement features."),
            (1.7, "Neighbour-label transformer stx_nb", LEARNED,
             "The same transformer with ADS-B modality dropout and one more head: a leave-one-out mean of the clipped residual of the other departures within \u00b130 min, per airport (nbm) and per runway (nbr), "
             "the queue level the flights share. It hands the stackers its prediction, gap to the chain, spread, P(label > 1,800 s), P(> 3,600 s), nbm30 and nbr30.")), 0),
        ("6", "Record tail", row(
            (1, "LIRF no-plan copy classifier", LEARNED,
             "CatBoost trained on all twelve months, each out of fold. For LIRF departures without an NM off-block that took off 1\u20135.6 h after schedule; applied with weight 0.5, ADS-B-matched rows only."),
            (1, "LIRF near-schedule rule", FITTED,
             "Two carriers' no-plan departures that take off within 1 h of schedule are set to the group's mean 2025 label."),
            (1, "Daily delay calibration", FITTED,
             "One multiplicative factor per airport-day makes the predicted total match EUROCONTROL's published daily delay; 0.6 of the change is applied. No per-flight value."),
            (1, "Duplicated-record rule", FITTED,
             "The NM-matched half of a duplicated take-off record gets a + b \u00b7 (MVT \u2212 LOBT), a line fitted on 2025 pairs outside the scored month.")), 0),
        ("7", "Four post-steps", row(
            (1, "LIRF copy re-mix", FITTED,
             "Refits the chain's implied copy weight per band of schedule delay and plan or no plan, on LIRF rows where the plan copy and the rules-chain value differ by over 600 s."),
            (1, "Runway-neighbour level: LightGBM", LEARNED,
             "Fits clip(pred \u2212 y, \u00b11,800 s) on 23 columns: shrunk means of the neighbours' predictions and clock gaps (runway \u00b15, \u00b115, \u00b130 min; airport \u00b115 min). Its output is subtracted."),
            (1, "Far-LIRF record mix", FITTED,
             "5\u20137 LIRF rows per phase with no NM off-block and plan copy above 50,000 s: p \u00b7 C + (1 \u2212 p) \u00b7 87,500, p the 2025 copy share at the same first-stage probability."),
            (1, "Arrival-twin window rule", RULE,
             "LIRF departures without an NM flight whose arrival twin has one: the prediction becomes the copy or the rules-chain value. No fitted parameter.")), 0),
        ("8", "Plan-window step", row((1, "Clip into the plan window", RULE,
            "Matched rows are clipped to MVT \u2212 LOBT \u00b1 3,606 s; some unmatched rows are moved relative to the window. 84 of the 670,790 test rows change. "
            "The result is the final prediction (CV 279.61 on the two folds).")), 0),
    ]
    PAD = 2.6
    heights = [max(chip_h(c) for c in chips) + extra + PAD for _, _, chips, extra in stages]
    H = sum(heights) + 1.0
    fig_h = H * 0.1056 + 1.2 + 0.4
    fig = plt.figure(figsize=(11, fig_h))
    ax = fig.add_axes([0.02, 0.4 / fig_h, 0.96, H * 0.1056 / fig_h])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, H)
    ax.axis("off")
    y = H - 0.5
    ys = []
    for (num, title, chips, extra), h in zip(stages, heights):
        top = y - PAD / 2
        ys.append(top - 1.2)
        ch = max(chip_h(c) for c in chips)
        x = X0
        for c in chips:
            col = c["col"]
            ax.add_patch(FancyBboxPatch((x, top - ch), c["w"], ch, boxstyle="round,pad=0,rounding_size=0.9", fc=col + "1c", ec=col, lw=1.4, zorder=3))
            ty = top - 1.0
            ax.text(x + 1.0, ty, c["title"] if len(c["title"]) < int(c["w"] * 1.25) else textwrap.fill(c["title"], int(c["w"] * 1.25)), fontsize=8.6, fontweight="bold", color=INK, va="top", zorder=4)
            nt = 1 + c["title"][:0].count("\n")
            ty -= 1.7 * (1 + (len(c["title"]) >= int(c["w"] * 1.25))) - (0.2 if len(c["title"]) < int(c["w"] * 1.25) else 0)
            ax.text(x + 1.0, ty, TAG[col], fontsize=6.9, color={BLUE: BLUE, AQUA: "#0e8a5f", MUTED: INK2, YELLOW: "#9a6a00"}[col], va="top", style="italic", fontweight="bold", zorder=4)
            ty -= 1.3
            ax.text(x + 1.0, ty, "\n".join(lines(c["text"], c["w"])), fontsize=7.6, color=INK2, va="top", linespacing=1.3, zorder=4)
            x += c["w"] + GAP
        if num == "2":  # stacked bar: share of the blend by family
            by = top - ch - 3.6
            x = X0
            for fam, colr in [("CatBoost", BLUE), ("MLP", ORANGE), ("Autoencoder", AQUA)]:
                w = W0 * wsum[fam] / wsum.sum()
                ax.add_patch(plt.Rectangle((x, by), w, 2.4, fc=colr, ec=SURFACE, lw=1.5, zorder=3))
                if fam != "Autoencoder":
                    ax.text(x + w / 2, by + 1.2, f"{fam} {wsum[fam]:.3f}", ha="center", va="center", fontsize=7.8, color="white", fontweight="bold", zorder=4)
                x += w
            ax.text(X0 + W0, by - 0.6, "Each fits non-copy rows only, as a correction to a clock anchor. Blend: non-negative least squares, one weight set for all airports (CV 429.50); weights sum to 0.998 as rounded.", ha="right", va="top", fontsize=7, color=MUTED)
        # stage label on the left
        ax.add_patch(Circle((2.4, top - 1.9), 1.55, fc=INK, ec="none", zorder=5))
        ax.text(2.4, top - 1.9, num, ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=6)
        ax.text(5.0, top - 0.8, textwrap.fill(title, 12), ha="left", va="top", fontsize=8.8, fontweight="bold", color=INK, linespacing=1.2)
        y -= h
        ax.plot([0, 100], [y, y], color=GRID, lw=0.8, zorder=1)
    ax.plot([2.4, 2.4], [ys[0] - 0.7, ys[-1] + 0.7], color=AXIS, lw=1.2, zorder=1)
    leg = fig.legend(handles=[Patch(fc=c + "55", ec=c, label=TAG[c]) for c in (LEARNED, FITTED, RULE, INPUT)], loc="upper right",
                     bbox_to_anchor=(0.98, 1 - 0.3 / fig_h), frameon=False, fontsize=8.5, ncol=4, handlelength=1.1, handleheight=1.1, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.02, 1 - 0.18 / fig_h, "Which model does what", fontsize=15, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.02, 1 - 0.62 / fig_h, "Each stage of the final chain, top to bottom: the learner, the job, and what it reads. Stages without a learner are marked as rules or fitted constants.",
             fontsize=10, ha="left", va="top", color=INK2)
    fig.text(0.02, 0.012, "Every number and name is from docs/model.md. Blue: a learned model. Green: constants or a calibration fitted on 2025 data, no learner. Grey: a fixed rule or formula. Yellow: a list of inputs.",
             fontsize=8, ha="left", va="bottom", color=MUTED)
    save(fig, "fig_models")


if __name__ == "__main__":
    fig_progress()
    fig_regimes()
    fig_months()
    fig_tail()
    fig_pipeline()
    fig_concentration()
    fig_timeline()
    fig_coverage()
    fig_airports()
    fig_mixture()
    fig_calibration()
    fig_rehearsal()
    fig_copy_reliability()
    fig_error_budget()
    fig_density()
    fig_hero()
    fig_taxi_schematic()
    fig_ensemble()
    fig_models()
