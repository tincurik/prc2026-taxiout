# SPDX-License-Identifier: GPL-3.0-or-later
"""Month-complete stacker (leave-two-months-out folds): price, then build.

The memb2 stackers learn from the Jan/Jul OOF only (344 k rows, 2 months), so they never see the other ten 2025 months' regimes
(LEMD / LIRF ADS-B coverage, snow days, summer peaks) with labels. The L2MO build has honest out-of-month base and member
predictions for all 12 months: Jan/Jul = each phase's base OOF, every other pair = the held-out test of its pseudo phase, each with
its own phase's features (memb2 gaps, ctx neighbours' base predictions). This trains the xgb stacker of adsb_stack.py (matched and
unmatched branches, same params, clip 1800, train_ymax 20000, LIRF uncorrected, seeds as adsb_stack) on those rows, two arms:
  jj   Jan/Jul rows only (today's stacker, same code)        m12   Jan/Jul rows + every other 2025 pair the phase may see
  final    Jan/Jul OOF = day5 folds over Jan/Jul days (m12: the five pseudo pairs in every training split);
           2026 test = one model on all the arm's rows, iterations = mean fold best x 1.1 (adsb_stack's test model)
  pseudoP  P's held-out pair = one model on phase P's Jan/Jul OOF rows (m12: + the four other pseudo pairs), early-stopped on a 15 %
           day split; with --phase_oof 1 also phase P's Jan/Jul OOF (day5, m12: + those four pairs) for its tail
Second-order leak, as in any stacking on OOF: pair Q's rows that train phase P's model come from members that saw P's labels.
Stacker types as adsb_stack.py: --model xgb (default, params max_depth 7), --covw 1 (the covw stacker: Jan/Jul rows weighted to the
2026 ADS-B coverage mix of their phase, other pairs weight 1 as adsb_stack's --aux), --model cat (CatBoost; GPU unless STACK_CAT_CPU),
--model lgb (pass the lgb2 --params); the test model copies adsb_stack's, including its unweighted xgb unmatched branch for xgb and cat.
Writes <id>_<arm> in outputs/overnight_<phase>/{oof,test}/ and reports/overnight_final/results/<id>_<arm>.json; score with
  python stack12.py --id s12ctx --extra adsbdir4,adsbrev,memb2,ctx [--seeds 1] [--arms jj,m12] [--phases final,pseudo2,...]
  python stack12.py --id s12w --covw 1 --arms m12 --seeds 3 --phase_oof 1      (one stacker of a full build, every phase)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adsb_stack as S  # noqa: E402
from ocommon import ROOT, report, rmse  # noqa: E402

PHASES = ["final", "pseudo2", "pseudo3", "pseudo4", "pseudo5", "pseudo6"]
ap = argparse.ArgumentParser()
ap.add_argument("--id", required=True)
ap.add_argument("--base", default="rc_v34_more18")
ap.add_argument("--extra", default="adsbdir4,adsbrev,memb2,ctx")
ap.add_argument("--feat", default="adsb2m")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--arms", default="jj,m12")
ap.add_argument("--phases", default=",".join(PHASES))
ap.add_argument("--phase_oof", type=int, default=0, help="1 = also each pseudo phase's Jan/Jul OOF (day5), needed for its tail")
ap.add_argument("--clip", type=float, default=1800)
ap.add_argument("--train_ymax", type=float, default=20000)
ap.add_argument("--skip", default="LIRF")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--model", default="xgb", choices=["xgb", "cat", "lgb"])
ap.add_argument("--covw", type=int, default=0, help="1 = the covw stacker (Jan/Jul rows weighted to the phase's 2026 coverage mix)")
ap.add_argument("--params", default=None, help="json overrides of adsb_stack's params (P for xgb/cat, LP for lgb); default xgb max_depth 7 "
                "as the base stackers, cat none (its unmatched test branch is xgb with adsb_stack's default P)")
ap.add_argument("--part", default="", help="suffix of the results json when the phases are split over jobs")
ap.add_argument("--final_phase", default=os.environ.get("EC_PHASE") or "final",
                help="directory name of the 'final' slot under this tree's data/processed_<p>, outputs/overnight_<p>")
ap.add_argument("--pairs_root", default=os.environ.get("PAIRS_ROOT") or None,
                help="tree holding the pseudo phases (their processed_<p>, overnight_<p>, data/<p>/truth.parquet); default this tree "
                     "(PAIRS_ROOT overrides it)")
ap.add_argument("--cap_id", default="", help="density-cap A/B (final phase only, overnight/density_cap.py): fit each arm's test model ONCE, write the predictions on the "
                "uncapped test features as <id>_<arm> and on the capped ones as <cap_id>_<arm>; needs --n_it_from, skips the day5 OOF (copy the shipped OOF)")
ap.add_argument("--cap_feat", default="adsb2mcap", help="capped twin of --feat (features_dep_<name>.parquet, the 2026 rows capped)")
ap.add_argument("--cap_dir", default="adsbdir4cap", help="capped twin of the adsbdir4 extra (adsb_direct.py --cap, test rows only)")
ap.add_argument("--ctl_dir", default="adsbdir4ctl", help="uncapped adsbdir4 test rows from the SAME adsb_direct fit as --cap_dir")
ap.add_argument("--n_it_from", default="", help="results json of a finished run of this stacker: its phases.final.<arm>.test_iters are used (no OOF refit)")
ap.add_argument("--smoke", type=int, default=0, help="1 = 3 %% of the rows, CPU xgb, outputs under /tmp/stack12_smoke")
a = ap.parse_args()
S.FEAT = a.feat
S.MODEL = a.model
(S.LP if a.model == "lgb" else S.P).update(json.loads(a.params) if a.params is not None else ({"max_depth": 7} if a.model == "xgb" else {}))
S.EXTRA.extend(x for x in a.extra.split(",") if x)
ARMS = a.arms.split(",")
RUN_PH = a.phases.split(",")
PSEUDO = [p for p in PHASES if p != "final"]
SKIP = [x for x in a.skip.split(",") if x]
PAIRS = Path(a.pairs_root).resolve() if a.pairs_root else ROOT
RUN = f"stack12-{a.id}{a.part}"
if a.smoke:
    S.P.update(device="cpu", nthread=2)
    S.LP.update(num_threads=2)
    RUN += "-smoke"
t0 = time.time()


def phroot(ph):  # the final slot lives in this tree, the pseudo phases (the L2MO pairs) in PAIRS
    return ROOT if ph == "final" else PAIRS


def phname(ph):
    return a.final_phase if ph == "final" else ph


def proc(ph):
    return phroot(ph) / f"data/processed_{phname(ph)}"


SMOKE = Path("/tmp/stack12_smoke")


def ind(ph):
    return phroot(ph) / f"outputs/overnight_{phname(ph)}"


def outd(ph):
    if a.smoke:
        for sub in ("oof", "test"):
            (SMOKE / ph / sub).mkdir(parents=True, exist_ok=True)
        return SMOKE / ph
    return phroot(ph) / f"outputs/overnight_{phname(ph)}"


APTS = None


def load(ph, kind, feat=None, dirx=None):
    """kind jj = the phase's Jan/Jul OOF rows, ho = a pseudo phase's held-out pair with truth, test = every row of the phase's test file."""
    S.PROC = proc(ph)
    ex0, ft0 = list(S.EXTRA), S.FEAT
    if dirx:
        S.EXTRA[:] = [dirx if x == "adsbdir4" else x for x in ex0]
    if feat:
        S.FEAT = feat
    try:
        return _load(ph, kind)
    finally:
        S.EXTRA[:], S.FEAT = ex0, ft0


def _load(ph, kind):
    if kind == "jj":
        o = pd.read_parquet(ind(ph) / "oof" / f"{a.base}.parquet")[["MVT_ID_mvt", "fold", "y", "pred"]]
    else:
        t = pd.read_parquet(ind(ph) / "test" / f"{a.base}.parquet").rename(columns={"TAXITIME_SEC_mvt": "pred"})
        if kind == "ho":
            tr = pd.read_parquet(PAIRS / f"data/{ph}/truth.parquet").rename(columns={"TAXITIME_SEC_mvt": "y"})
            tr["MVT_ID_mvt"], t["MVT_ID_mvt"] = tr.MVT_ID_mvt.astype("int64"), t.MVT_ID_mvt.astype("int64")
            o = tr.merge(t, on="MVT_ID_mvt", how="inner").assign(fold=ph)
            assert len(o) == len(tr), f"{ph}: base test misses truth rows"
        else:
            o = t.assign(y=np.nan, fold="test")
    if a.smoke:
        o = o[o.MVT_ID_mvt.astype("int64") % 33 == 0]  # same rows in the truth and test frames
    o = o.assign(pred0=o.pred, mult=1.0, imp=False, aux=False)
    d = S.frame(o[["MVT_ID_mvt", "fold", "y", "pred", "pred0", "mult", "imp", "aux"]])
    d["MVT_ID_mvt"] = d.MVT_ID_mvt.astype("int64")  # float64 in the test files; never downcast it
    for c in d.columns:
        if d[c].dtype == np.float64 and c not in ("y", "pred", "pred0"):
            d[c] = d[c].astype(np.float32)
    d["apt"] = pd.Categorical(d.apt.astype(str), categories=APTS)
    d["src"] = ph
    if a.covw:  # adsb_stack --covw: the phase's own Jan/Jul rows to its 2026 coverage mix; other pairs (aux) weight 1
        if kind == "jj":
            import covw_cv
            covw_cv.PROC = proc(ph)
            d["cw"] = covw_cv.cov_weights(d.MVT_ID_mvt.to_numpy())
        else:
            d["cw"] = 1.0
    return d.reset_index(drop=True)


def fit_es(tr, seed, kind):
    """adsb_stack.fit: early stopping on a 15 % day split of tr."""
    return S.fit(tr, a.clip, seed=seed, kind=kind)


def train_rows(d, kind):
    ok = d.ok.to_numpy()
    m = (ok if kind == "ok" else ~ok) & (d.y <= a.train_ymax).to_numpy() & ~d.imp.to_numpy()
    return d[m]


def apply_rows(d, kind):
    ok = d.ok.to_numpy()
    return (ok if kind == "ok" else ~ok) & ~d.apt.isin(SKIP).to_numpy()


def day5(jj, aux, label):
    """Jan/Jul OOF by day folds; aux rows (other pairs) join every training split. Returns new preds and the fold iterations."""
    new = jj.pred0.to_numpy().copy()
    days = np.sort(jj.day.unique())
    fk = jj.day.map(pd.Series(np.arange(len(days)) % a.k, index=days)).to_numpy()
    iters = {"ok": [], "un": []}
    for i in range(a.k):
        tr = jj[fk != i] if aux is None else pd.concat([jj[fk != i], aux], ignore_index=True)
        for kind in ("ok", "un"):
            va = (fk == i) & apply_rows(jj, kind)
            corr = np.zeros(int(va.sum()))
            for sd in range(a.seeds):
                m, it = fit_es(train_rows(tr, kind), sd, kind)
                iters[kind].append(it)
                corr += S.predict(m, jj.loc[va, S.feats(kind)], it) / a.seeds
            new[va] += corr
        print(f"  {label} fold k{i}: fit rows {len(tr):,}, iters ok {iters['ok'][-a.seeds:]} un {iters['un'][-a.seeds:]} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return new, iters


def full_model(tr, te, n_it, alts=()):
    """adsb_stack's test model: every training row, n_it[kind] iterations (no early stopping), seeds 100 + BAG_SEED + sd; weights (covw)
    on the matched branch; the unmatched branch is lgb for lgb and an unweighted xgb with P for xgb and cat (as adsb_stack.py).
    alts = more test frames (density-cap A/B): each fitted model predicts te and every alt, so the frames differ only in their features;
    returns the predictions of te, or of [te, *alts] when alts is given."""
    tes = [te, *alts]
    news = [t.pred0.to_numpy().copy() for t in tes]
    for kind in ("ok", "un"):
        f = train_rows(tr, kind)
        X, r = f[S.feats(kind)], np.clip(f.y - f.pred, -a.clip, a.clip)
        w = f["cw"].to_numpy() if "cw" in f else None
        vas = [apply_rows(t, kind) for t in tes]
        Xts = [t.loc[v, S.feats(kind)] for t, v in zip(tes, vas)]
        tree = S.MODEL == "lgb" or (S.MODEL == "cat" and kind == "ok")
        if not tree:
            dm = xgb.DMatrix(X, r, weight=w if kind == "ok" else None, enable_categorical=True)
            dts = [xgb.DMatrix(x, enable_categorical=True) for x in Xts]
        for sd in range(a.seeds):
            seed = 100 + S.BAG_SEED + sd
            if S.MODEL == "lgb":
                m = S.lgb_fit(X, r, w, seed, n_it[kind])
            elif tree:
                m = S.cat_fit(X, r, w, seed, n_it[kind])
            else:
                S.P["seed"] = seed
                m = xgb.train(S.P, dm, n_it[kind])
            for k in range(len(tes)):
                p = m.predict(dts[k]) if not tree else S.predict(m, Xts[k], n_it[kind])
                news[k][vas[k]] += p / a.seeds
    return news if alts else news[0]


def es_model(tr, te):
    """One early-stopped model per branch and seed on tr, applied to te."""
    new = te.pred0.to_numpy().copy()
    its = {}
    for kind in ("ok", "un"):
        va = apply_rows(te, kind)
        corr = np.zeros(int(va.sum()))
        its[kind] = []
        for sd in range(a.seeds):
            m, it = fit_es(train_rows(tr, kind), sd, kind)
            its[kind].append(it)
            corr += S.predict(m, te.loc[va, S.feats(kind)], it) / a.seeds
        new[va] += corr
    return new, its


def write_test(ph, te, new, name):
    tm = pd.read_parquet((PAIRS / f"data/{ph}/raw" if ph.startswith("pseudo") else ROOT / "data/raw") / "final_submitting.parquet")
    if a.smoke:
        tm = tm[tm.MVT_ID_mvt.astype("int64").isin(te.MVT_ID_mvt.astype("int64"))].copy()
    tm["TAXITIME_SEC_mvt"] = tm.MVT_ID_mvt.astype("int64").map(pd.Series(np.clip(new, 0, None), index=te.MVT_ID_mvt.astype("int64").to_numpy())).to_numpy()
    assert tm.TAXITIME_SEC_mvt.notna().all(), f"{ph}: {int(tm.TAXITIME_SEC_mvt.isna().sum())} template rows without a prediction"
    path = outd(ph) / "test" / f"{name}.parquet"
    assert not path.exists(), f"{path} exists (never overwrite)"
    tm.to_parquet(path, index=False)


def write_oof(ph, jj, new, name):
    path = outd(ph) / "oof" / f"{name}.parquet"
    assert not path.exists(), f"{path} exists (never overwrite)"
    pd.DataFrame({"MVT_ID_mvt": jj.MVT_ID_mvt, "fold": jj.fold, "y": jj.y, "pred": new}).to_parquet(path, index=False)


# ---------------------------------------------------------------- load
STEPS = 1 + len(RUN_PH) * len(ARMS)
report(RUN, 0, STEPS, "loading", purpose="Does training the stacker on all 12 months of 2025 (L2MO out-of-month rows) instead of Jan/Jul only cut RMSE on held-out months?")
for ph in RUN_PH:
    for arm in ARMS:
        for sub in ("oof", "test"):
            p = outd(ph) / sub / f"{a.id}_{arm}.parquet"
            if p.exists():
                raise SystemExit(f"{p} exists: a rerun gets a new --id")
APTS = sorted(pd.read_parquet(proc("final") / "features_dep.parquet", columns=["apt"]).apt.dropna().astype(str).unique())
HO = {q: load(q, "ho") for q in PSEUDO}
cols0 = list(HO[PSEUDO[0]].columns)
for q in PSEUDO:
    assert list(HO[q].columns) == cols0, f"{q}: feature columns differ"
print(f"held-out pairs loaded: { {q: len(v) for q, v in HO.items()} } ({(time.time() - t0) / 60:.1f} min)", flush=True)
report(RUN, 1, STEPS, "pairs loaded")
res = {"id": a.id, "base": a.base, "extra": a.extra, "feat": a.feat, "seeds": a.seeds, "stack_model": a.model, "covw": a.covw,
       "bag_seed": S.BAG_SEED, "clip": a.clip, "train_ymax": a.train_ymax,
       "skip": SKIP, "k": a.k, "params": S.LP if a.model == "lgb" else S.P, "phases": {}}
step = 1
for ph in RUN_PH:
    jj = load(ph, "jj")
    assert list(jj.columns) == cols0, f"{ph}: Jan/Jul feature columns differ"
    others = [q for q in PSEUDO if q != ph]
    aux = pd.concat([HO[q] for q in others], ignore_index=True)
    res["phases"][ph] = {}
    for arm in ARMS:
        name = f"{a.id}_{arm}"
        r = {"aux_rows": int(len(aux)) if arm == "m12" else 0}
        if ph == "final":
            if a.cap_id:  # density-cap A/B: one fit per arm, two sets of test features, no OOF (it does not depend on the 2026 features)
                n_it = json.load(open(a.n_it_from))["phases"]["final"][arm]["test_iters"]
                te, tec = load(ph, "test", dirx=a.ctl_dir), load(ph, "test", feat=a.cap_feat, dirx=a.cap_dir)
                assert (te.MVT_ID_mvt.to_numpy() == tec.MVT_ID_mvt.to_numpy()).all()
                tr = jj if arm == "jj" else pd.concat([jj, aux], ignore_index=True)
                t0n, t1n = full_model(tr, te, n_it, alts=[tec])
                write_test(ph, te, t0n, name)
                write_test(ph, te, t1n, f"{a.cap_id}_{arm}")
                chg = np.abs(t1n - t0n) > 1e-9
                r.update(test_iters=n_it, test_rms_change=float(np.sqrt(np.mean((t0n - te.pred0.to_numpy()) ** 2))),
                         cap_rms_change=float(np.sqrt(np.mean((t1n - t0n) ** 2))), cap_rows_changed=int(chg.sum()), cap_mean_shift=float(np.mean(t1n - t0n)))
                del te, tec
                print(f"{ph} {arm}: cap vs ctl rms {r['cap_rms_change']:.2f} s, mean {r['cap_mean_shift']:+.2f} s, rows moved {r['cap_rows_changed']:,} "
                      f"(stacker rms change vs chain {r['test_rms_change']:.1f})", flush=True)
            else:
                new, iters = day5(jj, aux if arm == "m12" else None, f"{ph} {arm}")
                y = jj.y.to_numpy()
                r.update(oof_rmse=rmse(new, y), oof_base=rmse(jj.pred0, y), oof_clean=rmse(new[y <= 7200], y[y <= 7200]),
                         oof_by_fold={f: rmse(new[(jj.fold == f).to_numpy()], y[(jj.fold == f).to_numpy()]) for f in ("jan", "jul")})
                write_oof(ph, jj, new, name)
                te = load(ph, "test")
                n_it = {k: int(np.mean(v) * 1.1) for k, v in iters.items()}
                tr = jj if arm == "jj" else pd.concat([jj, aux], ignore_index=True)
                tnew = full_model(tr, te, n_it)
                write_test(ph, te, tnew, name)
                r.update(test_iters=n_it, test_rms_change=float(np.sqrt(np.mean((tnew - te.pred0.to_numpy()) ** 2))))
                del te
                print(f"{ph} {arm}: OOF {r['oof_rmse']:.2f} (base {r['oof_base']:.2f}) jan {r['oof_by_fold']['jan']:.2f} "
                      f"jul {r['oof_by_fold']['jul']:.2f}; 2026 test rms change {r['test_rms_change']:.1f}", flush=True)
        else:
            ho = HO[ph]
            tr = jj if arm == "jj" else pd.concat([jj, aux], ignore_index=True)
            te = load(ph, "test")
            if a.phase_oof:  # the pipeline's procedure: day5 OOF for the phase's tail, then one model on every row
                onew, iters = day5(jj, aux if arm == "m12" else None, f"{ph} {arm} oof")
                write_oof(ph, jj, onew, name)
                r["oof_rmse"] = rmse(onew, jj.y.to_numpy())
                its = {k: int(np.mean(v) * 1.1) for k, v in iters.items()}
                tnew = full_model(tr, te, its)
            else:
                tnew, its = es_model(tr, te)
            write_test(ph, te, tnew, name)
            hp = np.clip(pd.Series(tnew, index=te.MVT_ID_mvt.astype("int64").to_numpy())
                         .reindex(ho.MVT_ID_mvt.astype("int64").to_numpy()).to_numpy(), 0, None)
            del te
            y = ho.y.to_numpy()
            mon = pd.to_datetime(ho.MVT_TIME_UTC_mvt, unit="s", utc=True).dt.month.to_numpy()
            r.update(ho_rmse=rmse(hp, y), ho_base=rmse(ho.pred0, y), ho_clean=rmse(hp[y <= 7200], y[y <= 7200]), iters=its,
                     by_month={int(m): rmse(hp[mon == m], y[mon == m]) for m in np.unique(mon)})
            print(f"{ph} {arm}: held-out {r['ho_rmse']:.2f} (base {r['ho_base']:.2f}) by month "
                  f"{ {m: round(v, 2) for m, v in r['by_month'].items()} } iters {its}", flush=True)
        res["phases"][ph][arm] = r
        step += 1
        report(RUN, step, STEPS, f"{ph} {arm} done ({(time.time() - t0) / 60:.1f} min)")
    del jj, aux
res["minutes"] = round((time.time() - t0) / 60, 1)
RES = ROOT / f"reports/overnight_{a.final_phase}"
(RES / "results").mkdir(parents=True, exist_ok=True)
out = (SMOKE if a.smoke else RES / "results") / f"{a.id}{a.part}.json"
out.write_text(json.dumps(res, indent=1, default=str))
with open(SMOKE / "results.jsonl" if a.smoke else RES / "results.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"id": a.id, "model": "stack12", **{k: v for k, v in res.items() if k not in ("id", "params")}}, default=str) + "\n")
print(json.dumps(res, indent=1, default=str)[:4000], flush=True)
report(RUN, STEPS, STEPS, f"done in {res['minutes']} min", finished=True)
