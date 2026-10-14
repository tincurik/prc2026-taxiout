# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the interactive write-up writeup/index.html from writeup/data/*.csv, the tables in docs/ and story_template.html.

    python writeup/build_story.py        # CPU, under a second; no model is run, nothing is downloaded

Every number on the page comes from a CSV in writeup/data/ or from a table or sentence in docs/ (or the result table of the
root README.md). The template holds the prose; a number in prose is a placeholder {{key}} filled from FACTS below. A missing
table, column or sentence stops the build with the name of what is missing, so a doc edit cannot silently leave a stale number.
The only hand-set values are the two schematic flights (SCHEMATIC), drawn as illustrations exactly like make_figures.py's
fig_timeline and fig_mixture, and labelled schematic on the page.
"""
import csv
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
DOCS = ROOT / "docs"

# Illustrations, not data: the same schematic flights as fig_timeline / fig_mixture in figures/make_figures.py.
SCHEMATIC = {
    "ordinary": {"block": 14, "aobt": 16, "mvt": 30},   # minutes after SCHED
    "copy": {"aobt": 165, "mvt": 182},
    "mix": {"copy": 180, "phys": 15, "p": 0.3},
}


# ---------------------------------------------------------------- readers
def read_csv(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(s):
    return float(s) if s not in ("", None) else None


def md_tables(path):
    """All pipe tables of a Markdown file as lists of dicts keyed by the header cells."""
    out, cur = [], None
    for line in (path.read_text(encoding="utf-8")).splitlines():
        if line.startswith("|"):
            cells = [c.strip().replace("\x00", "|") for c in line.replace("\\|", "\x00").strip().strip("|").split("|")]
            if cur is None:
                cur = {"head": cells, "rows": []}
            elif all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            else:
                cur["rows"].append(dict(zip(cur["head"], cells)))
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def md_table(path, first_head):
    """The table of a doc whose first header cell is first_head."""
    for t in md_tables(path):
        if t["head"][0].strip("* ").lower() == first_head.lower():
            return t["rows"]
    raise SystemExit(f"build_story: no table starting with '{first_head}' in {path.relative_to(ROOT)}")


def doc_num(path, pattern, group=1):
    """A number quoted in a doc's prose, found by a regex; stops the build when the sentence has changed."""
    text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"build_story: pattern {pattern!r} not found in {path.relative_to(ROOT)}")
    return m.group(group)


def md_inline(s):
    """Markdown cell text to safe HTML: bold, code, links (to the docs pages), escaped otherwise."""
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<a href="{doc_link(m.group(2))}">{m.group(1)}</a>', s)
    return s


def doc_link(target):
    if target.startswith(("http://", "https://", "#")):
        return target
    return "../docs/" + target.split("/")[-1]


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def months_of(text):
    """'Jan–May and Sep–Dec 2025' -> [1, 2, 3, 4, 5, 9, 10, 11, 12]."""
    text = re.sub(r"\(.*?\)|\d{4}", "", text)
    out = []
    for part in re.split(r",| and ", text):
        names = re.findall(r"[A-Z][a-z]{2}", part)
        if "–" in part and len(names) == 2:
            a, b = MONTHS.index(names[0]), MONTHS.index(names[1])
            out += list(range(a + 1, b + 2))
        else:
            out += [MONTHS.index(n) + 1 for n in names]
    return out


def fnum(x, nd=0):
    s = f"{x:,.{nd}f}"
    return s.replace("-", "−")


def plain_num(s):
    return float(re.sub(r"[^\d.\-−]", "", s).replace("−", "-"))


# ---------------------------------------------------------------- data
def build():
    progress = read_csv("progress.csv")
    regimes = read_csv("label_regimes.csv")
    sse = read_csv("sse_curve.csv")
    tail = read_csv("tail_steps.csv")
    reh = read_csv("rehearsal.csv")
    apt = read_csv("airport_error.csv")
    months = read_csv("month_error.csv")
    cov = read_csv("adsb_coverage.csv")
    cal = read_csv("calibration.csv")
    budget = read_csv("error_budget.csv")
    bulk = read_csv("error_bulk_cells.csv")
    oracle = {r["window_min"]: float(r["gain_rmse_s"]) for r in read_csv("level_oracle.csv")}
    ebd = DOCS / "error-budget.md"
    hubs = [r for r in bulk if r["apt"] in ("LTFM", "LFPG", "EGLL", "LIRF") and r["adsb"] == "no_adsb"]
    hub_share = fnum(sum(float(r["share_pct"]) for r in hubs), 1)
    if doc_num(ebd, r"without a track hold ([\d.]+ %) of all") != hub_share + " %":
        raise SystemExit("build_story: the hub share in docs/error-budget.md no longer matches error_bulk_cells.csv")
    art = [r for r in budget if r["kind"] == "artefact"]

    # draft flags of the progress steps come from the results table, matched on the CV after the step
    res_prog = md_table(DOCS / "results.md", "stage")
    draft_after = set()
    for r in res_prog:
        if "draft" in r["stage"] and "→" in r["CV before → after"]:
            draft_after.add(round(plain_num(r["CV before → after"].split("→")[1]), 2))
    prog = []
    for r in progress:
        a, b = float(r["cv_from"]), float(r["cv_to"])
        prog.append({"stage": r["stage"], "family": r["family"], "from": a, "to": b, "draft": round(b, 2) in draft_after})

    mech = [{"name": md_inline(r["mechanism"]), "evidence": md_inline(r["evidence in 2025"]), "use": md_inline(r["how the model uses it"])}
            for r in md_table(DOCS / "task-and-data.md", "mechanism") if "evidence in 2025" in r]
    taxonomy = None
    for t in md_tables(DOCS / "task-and-data.md"):
        if t["head"][:2] == ["mechanism", "rows"]:
            taxonomy = [{"name": md_inline(r["mechanism"]), "rows": r["rows"], "cost": plain_num(r["share of the cost"])} for r in t["rows"]]
    if taxonomy is None:
        raise SystemExit("build_story: taxonomy table missing in task-and-data.md")

    families = [{"family": re.sub(r"\*", "", r["family"]).replace("(draft)", "").strip(), "draft": "draft" in r["family"],
                 "holds": md_inline(r["what it holds"]), "why": md_inline(r["operational reason"])}
                for r in md_table(DOCS / "features.md", "family")]

    members = [{"n": r["member"], "learner": md_inline(r["learner"]), "anchor": r["anchor"], "inputs": md_inline(r["inputs beyond the base set"]),
                "cv": plain_num(r["CV RMSE alone, as a mixture"]), "w": plain_num(r["blend weight"])}
               for r in md_table(DOCS / "model.md", "member")]
    blend = {r["blend"]: plain_num(r["CV RMSE"]) for r in md_table(DOCS / "model.md", "blend")}
    stackers = [{"name": md_inline(r["stacker"]), "cv": plain_num(r["CV RMSE of the chain after it"])} for r in md_table(DOCS / "model.md", "stacker")]
    tail_doc = {re.sub(r"\W+", " ", r["step"]).strip().lower(): md_inline(r["what it does"]) for r in md_table(DOCS / "model.md", "step")}
    tails = []
    for r in tail:
        key = re.sub(r"\W+", " ", r["step"]).strip().lower()
        tails.append({"step": r["step"], "cv": float(r["cv_rmse"]), "d": num(r["delta"]), "lo": num(r["ci_low"]), "hi": num(r["ci_high"]),
                      "what": tail_doc.get(key, "")})
    reh_steps = []
    for r in md_table(DOCS / "validation.md", "step"):
        def parse(c):
            m = re.match(r"([\d.]+) → ([\d.]+), ([−+\-\d,]+) \[([−+\-\d,]+), ([−+\-\d,]+)\]", c)
            if not m:
                return None
            return [float(m.group(1)), float(m.group(2))] + [float(m.group(i).replace(",", "").replace("−", "-")) for i in (3, 4, 5)]
        reh_steps.append({"step": r["step"], "feb": parse(r[[k for k in r if k.startswith("Feb")][0]]),
                          "jun": parse(r[[k for k in r if k.startswith("Jun")][0]])})
    matched = [{"rows": r["rows"], "count": r["count"], "chain": r["after the rules chain: RMSE / ≤ 2 h"],
                "stack": r["after the stackers"], "ref": r["reference chain"]} for r in md_table(DOCS / "results.md", "rows")]

    folds = []
    for r in md_table(DOCS / "validation.md", "fold"):
        folds.append({"fold": r["fold"], "val": months_of(r["validation month"]), "train": months_of(r["training months"]),
                      "buffer": months_of(r["buffer months (left out of training)"])})
    n_cv = sum(int(r["rows"]) for r in regimes)
    ref = float(tail[-1]["cv_rmse"])
    rows_by = {int(r["rows"]): r for r in sse}

    def sse_at(k):
        return float(rows_by[k]["share_sse"]) * 100

    reh_all = {r["month"]: r for r in reh if r["apt"] == "all"}
    final_row = next(r for r in md_tables(ROOT / "README.md")[0]["rows"] if "Final model" in r[""] or "Final model" in "".join(r.values()))
    fm = re.search(r"Jan ([\d.]+), Jul ([\d.]+)", "".join(final_row.values()))
    assert fm, "README headline row must read 'Jan X, Jul Y'"
    reg = {r["regime"]: r for r in regimes}
    phys, copy_r = reg["Physical taxi (up to 1 h)"], reg["Schedule copy"]
    vlong = reg["Very long, over 2 h"]

    tad = DOCS / "task-and-data.md"
    mdl = DOCS / "model.md"
    clf = {r["rows"]: r for r in md_table(mdl, "rows")}
    F = {
        "baseline": fnum(float(progress[0]["cv_from"]), 2),
        "final": fnum(float(progress[-1]["cv_to"]), 2),
        "final_jan": fm.group(1), "final_jul": fm.group(2) or fm.group(1),
        "cut_pct": fnum(100 * (1 - float(progress[-1]["cv_to"]) / float(progress[0]["cv_from"])), 0),
        "ref": fnum(ref, 2),
        "n_cv": fnum(n_cv),
        "train_rows": doc_num(tad, r"([\d,]+) labelled departures"),
        "test_rows": doc_num(tad, r"\| Test \| ([\d,]+) departures"),
        "copy_share_2025": doc_num(tad, r"\*\*Schedule copy\.\*\*[^|]*\| ([\d.]+ %) of 2025"),
        "copy_month_range": doc_num(tad, r"of 2025 departures \(([\d.–]+ %) in every month\)"),
        "lirf_copy": doc_num(tad, r"LIRF (\d+ %), other airports"),
        "phys_rows": fnum(100 * float(phys["share_rows"]), 0), "phys_sse": fnum(100 * float(phys["share_sse"]), 0),
        "copy_rows": fnum(100 * float(copy_r["share_rows"]), 0), "copy_sse": fnum(100 * float(copy_r["share_sse"]), 0),
        "vlong_n": fnum(int(vlong["rows"])), "vlong_sse": fnum(100 * float(vlong["share_sse"]), 0),
        "top2": fnum(sse_at(2), 1), "top10": fnum(sse_at(10), 1),
        "top100": doc_num(tad, r"10 rows [\d.]+ %, 100 rows ([\d.]+ %)"),
        "top1000": doc_num(tad, r"1,000 rows ([\d.]+ %)"),
        "without2": doc_num(tad, r"RMSE would be ([\d.]+) s instead of"),
        "top2_labels": doc_num(tad, r"\(labels ([\d,]+ s and [\d,]+ s)\)"),
        "top2_hours": doc_num(tad, r"lies ([\d.]+ h and [\d.]+ h) \*before\*"),
        "plan_cliff": doc_num(tad, r"(0 copies among [\d,]+ departures)"),
        "plan_window": doc_num(tad, r"`\\\|BLOCK − LOBT\\\| ≤ ([\d,]+ s)`"),
        "stand_prev_copy": doc_num(DOCS / "features.md", r"copies in ([\d.]+ %) of cases when the stand's previous arrival was one"),
        "stand_prev_else": doc_num(DOCS / "features.md", r"was one, against ([\d.]+ %) otherwise"),
        "first_heard": doc_num(DOCS / "features.md", r"\(median (\+\d+ s), transponders"),
        "logloss": clf["all"]["log loss (constant at the base rate)"].split(" ")[0],
        "logloss_const": clf["all"]["log loss (constant at the base rate)"].split("(")[1].rstrip(")"),
        "clf_auc": clf["all"]["AUC"], "clf_share": clf["all"]["copy share"], "clf_meanp": clf["all"]["mean `p`"],
        "blend_cv": fnum(blend[next(k for k in blend if "cross-fitted" in k)], 2),
        "blend_eq": fnum(blend["equal weights"], 2),
        "rules_cv": fnum(float(tail[0]["cv_rmse"]), 2),
        "stack_cv": fnum(float(tail[1]["cv_rmse"]), 2),
        "dly_corr": doc_num(mdl, r"\(correlation ([\d.]+), median ratio"),
        "dly_apply": doc_num(mdl, r"applying ([\d.]+) of that change"),
        "dly_gain": doc_num(DOCS / "daily-delay-file.md", r"reference chain \(\[results\]\(results\.md\)\) \| [\d.]+ \| [\d.]+ \| \+([\d.]+)"),
        "disagree": doc_num(mdl, r"it gained (−[\d.]+ CV RMSE \[[^\]]+\])"),
        "disagree_reh": doc_num(mdl, r"and (−[\d.]+ \[[^\]]+\]) on the held-out rehearsal months"),
        "ctx_gain": doc_num(mdl, r"weighted like the test months \(\[validation\]\([^)]*\)\), (−[\d.]+ RMSE \[[^\]]+\])"),
        "m12_gain": doc_num(mdl, r"Judged like the context features: (−[\d.]+ RMSE \[[^\]]+\])"),
        "m12_rows": doc_num(mdl, r"\(([\d.]+ M) rows\)"),
        "reh": fnum(float(reh_all["all"]["rmse"]), 2),
        "reh_feb": fnum(float(reh_all["2025-02"]["rmse"]), 2), "reh_jun": fnum(float(reh_all["2025-06"]["rmse"]), 2),
        "reh_rows": fnum(int(reh_all["all"]["n"])),
        "reh_clean": fnum(float(reh_all["all"]["rmse_le2h"]), 2),
        "test_weights": doc_num(DOCS / "validation.md", r"share of the test rows: (.+? Jul)\."),
        "pairs": doc_num(DOCS / "validation.md", r"four more pairs rerun the whole pipeline: ([^.]+?) 2025\."),
        "miss_cost": doc_num(mdl, r"costs about (10⁸ s²)"),
        "auc_years": doc_num(DOCS / "validation.md", r"(AUC [\d.]+), but Jan/Jul 2025"),
        "auc_same": doc_num(DOCS / "validation.md", r"Jan/Jul 2025 from Jan/Jul 2026 only at ([\d.]+)"),
        "mix_copy": SCHEMATIC["mix"]["copy"], "mix_phys": SCHEMATIC["mix"]["phys"],
        "member_best": fnum(min(plain_num(r["CV RMSE alone, as a mixture"]) for r in md_table(mdl, "member")), 2),
        "member_worst": fnum(max(plain_num(r["CV RMSE alone, as a mixture"]) for r in md_table(mdl, "member")), 2),
        "pool_rows": doc_num(ebd, r"([\d,]+) departures, scored"),
        "pool_rmse": doc_num(ebd, r"RMSE on this pool is ([\d.]+) s"),
        "pool_rmse_noart": doc_num(ebd, r"Without them the RMSE is ([\d.]+) s instead of"),
        "art_rows": fnum(sum(int(r["rows"]) for r in art)),
        "art_share": doc_num(ebd, r"[\d,]+ rows hold ([\d.]+) % of the squared error"),
        "bulk_rows": fnum(int(next(r["rows"] for r in budget if r["key"] == "bulk"))),
        "bulk_share": next(r["share_pct"] for r in budget if r["key"] == "bulk"),
        "hub_share": hub_share,
        "oracle15": fnum(oracle["15"], 2), "oracle30": fnum(oracle["30"], 2), "oracle60": fnum(oracle["60"], 2),
        "bundle_gain": doc_num(mdl, r"against the context chain: (−[\d.]+ RMSE \[[^\]]+\])"),
        "post_gain": doc_num(mdl, r"All four together\s+\*\*(−[\d.]+ \[[^\]]+\])\*\*"),
        "bundle_cv": doc_num(mdl, r"On the January/July folds the final chain goes ([\d.]+) →"),
        "nb_gain": doc_num(mdl, r"with `ts_kdu3`: (−[\d.]+ RMSE \[[^\]]+\])"),
        "copy_sse_pct": fnum(100 * float(copy_r["share_sse"]), 1),
        # scorecard and anchor
        "reh_base": doc_num(DOCS / "validation.md", r"baseline on the same rows scores ([\d.]+)"),
        "anchor_to": fnum(float(progress[0]["cv_to"]), 2), "anchor_gain": fnum(float(progress[0]["cv_from"]) - float(progress[0]["cv_to"]), 0),
        "n_aobt": str(sum(1 for r in md_table(mdl, "member") if r["anchor"].strip() == "AOBT")),
        "n_lobt": str(sum(1 for r in md_table(mdl, "member") if r["anchor"].strip() == "LOBT")),
        # the monthly runway calibration
        "runway_gain": fnum(-next(float(r["delta"]) for r in tail if r["step"].startswith("Monthly runway")), 2),
        "runway_ci": next(f'{fnum(float(r["ci_low"]), 2)}, {fnum(float(r["ci_high"]), 2)}' for r in tail if r["step"].startswith("Monthly runway")),
        "runway_cap": doc_num(mdl, r"a quarter of the way towards the published mean plus offset \(at most (\d+) s per cell\)"),
        "runway_pairs": doc_num(mdl, r"it gains in every held-out pair \((\d of \d)\)"),
        "runway_cost": doc_num(mdl, r"That costs \*\*(\+[\d.]+ RMSE \[[^\]]+\])\*\*"),
        "airport_names": ", ".join(f"{k} {v}" for k, v in D_names(tad).items()),
    }
    D = {
        "progress": prog, "regimes": regimes, "sse": [[int(r["rows"]), float(r["share_sse"]), float(r["rmse_without"])] for r in sse],
        "ref": ref, "n_cv": n_cv, "mech": mech, "taxonomy": taxonomy, "families": families, "members": members,
        "stackers": stackers, "tail": tails, "reh_steps": reh_steps, "matched": matched,
        "rehearsal": [{k: (v if k in ("month", "apt") else float(v)) for k, v in r.items()} for r in reh],
        "airports": [{k: (v if k in ("apt", "fold") else float(v)) for k, v in r.items()} for r in apt],
        "months": [{k: float(v) for k, v in r.items()} for r in months],
        "coverage": {"cols": list(cov[0].keys())[1:], "rows": [{"apt": r["apt"], "v": [float(r[c]) for c in list(r)[1:]]} for r in cov]},
        "calibration": [{"view": r["view"], "band": r["band"], "n": int(r["n"]), "bias": float(r["bias"]), "rmse": float(r["rmse"])} for r in cal],
        "schematic": SCHEMATIC, "folds": folds, "anchor": float(progress[0]["cv_to"]),
        "budget": [{"key": r["key"], "label": r["label"], "kind": r["kind"], "rows": int(r["rows"]), "share": float(r["share_pct"]), "rmse": float(r["rmse"])} for r in budget],
        "bulk": [{"apt": r["apt"], "adsb": r["adsb"], "rows": int(r["rows"]), "share": float(r["share_pct"]), "rmse": float(r["rmse"])} for r in bulk],
        "oracle": [{"win": int(k), "gain": v} for k, v in oracle.items()],
        # the overview figure: CV RMSE after each box of the pipeline (Jan + Jul 2025 folds)
        "pipe": {"baseline": float(progress[0]["cv_from"]), "anchor": float(progress[0]["cv_to"]), "mix": blend[next(k for k in blend if "cross-fitted" in k)],
                 "rules": float(tail[0]["cv_rmse"]), "stack": float(tail[1]["cv_rmse"]), "ref": ref,
                 "final": float(progress[-1]["cv_to"]), "final_draft": prog[-1]["draft"],
                 "logloss": clf["all"]["log loss (constant at the base rate)"], "auc": clf["all"]["AUC"],
                 "members": [min(m["cv"] for m in members), max(m["cv"] for m in members)]},
        "reliability": [{"group": r["group"], "bin": r["bin"], "n": int(r["n"]), "p": float(r["mean_p"]), "rate": float(r["copy_rate"])}
                        for r in read_csv("copy_calibration.csv")] if (DATA / "copy_calibration.csv").exists() else [],
        "names": {r.split(" ", 1)[0]: r.split(" ", 1)[1] for r in
                  re.findall(r"([A-Z]{4} \([^)]+\))", doc_num(tad, r"The ten airports are ([^.]+)\.", 1).replace(" and ", ", "))},
    }
    D["names"] = {k: v.strip("()") for k, v in D["names"].items()}
    return D, F


def D_names(tad):
    names = {r.split(" ", 1)[0]: r.split(" ", 1)[1].strip("()") for r in
             re.findall(r"([A-Z]{4} \([^)]+\))", doc_num(tad, r"The ten airports are ([^.]+)\.", 1).replace(" and ", ", "))}
    return names


def main():
    D, F = build()
    tpl = (HERE / "story_template.html").read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(F))
    if missing:
        raise SystemExit(f"build_story: template uses unknown facts {missing}")
    page = re.sub(r"\{\{(\w+)\}\}", lambda m: html.escape(str(F[m.group(1)]), quote=False), tpl)
    page = page.replace("/*DATA*/null", json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    import base64
    alts = {"fig_taxi_schematic": "Illustration of one departure from stand to take-off, with the data sources that see each part of it.",
            "fig_hero": "Left: the RMSE after each stage, from 661 s to 280 s. Right: predicted against recorded taxi-out time for 344,419 departures.",
            "fig_density": "Predicted against recorded taxi-out time, and the distribution of the prediction error."}

    def figure(m):  # an offline, self-contained picture: the SVG written by figures/make_figures.py as a data URI
        svg = (HERE / "figures" / f"{m.group(1)}.svg").read_bytes()
        return f'<img alt="{alts[m.group(1)]}" src="data:image/svg+xml;base64,{base64.b64encode(svg).decode()}">'
    page = re.sub(r"<!--SFIG (\w+)-->", figure, page)
    (HERE / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote writeup/index.html ({len(page) / 1024:.0f} kB, {len(F)} facts)")


if __name__ == "__main__":
    main()
