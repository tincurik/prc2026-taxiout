# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare two chains the way the write-up does: per held-out month, then weighted by the test months' row shares.

Held-out rows (no fit saw them; the two chains are paired inside each phase):
  final    Jan + Jul 2025   outputs/overnight_final/oof/<id>.parquet            (columns MVT_ID_mvt, y, pred)
  pseudo2  Feb + Jun 2025   outputs/overnight_pseudo2/test/<id>.parquet         against data/pseudo2/truth.parquet
  pseudo3  Mar + Sep        pseudo4  Apr + Oct        pseudo5  May + Nov        pseudo6  Aug + Dec
For every month m: x_m = the change in MSE (candidate - reference) and SE_m from a paired bootstrap over whole days (days resampled inside the
month, 2000 draws), on all rows and on trim rows (label <= 20,000 s).  The between-month SD tau is the DerSimonian-Laird estimate; the
estimate for each test month T is shrunk towards the random-effects mean mu of all observed months,
theta_T = lam_T x_T + (1 - lam_T) mu with lam_T = tau^2 / (tau^2 + SE_T^2), and the final-months effect is E = sum_T share_T theta_T over
Jan .228, Feb .215, Jun .271, Jul .286 (the share of the test rows; needs the months 1, 2, 6, 7 to be observed).  One RMSE point on the test
months is about 535 MSE.  A change counts as a gain when E < 0 with its 95 % interval below zero and no test month with its interval wholly above zero.

  python overnight/month_weighted.py <candidate id> <reference id> [--phases final pseudo2 pseudo3 pseudo4 pseudo5 pseudo6] [--refit_sd 0]
--refit_sd adds the noise of refitting (in MSE) to every month's SE in quadrature (a change that refits a model should pass the seed noise).
"""
import argparse

import numpy as np
import pandas as pd

from ocommon import ROOT

FOLDS = {"final": (1, 7), "pseudo2": (2, 6), "pseudo3": (3, 9), "pseudo4": (4, 10), "pseudo5": (5, 11), "pseudo6": (8, 12)}
SHARE = {1: 0.228, 2: 0.215, 6: 0.271, 7: 0.286}
TRIM = 20000
B = 2000


def pred_path(phase, i):
    return ROOT / (f"outputs/overnight_final/oof/{i}.parquet" if phase == "final" else f"outputs/overnight_{phase}/test/{i}.parquet")


def day_month(phase):
    f = pd.read_parquet(ROOT / f"data/processed_{phase}/features_dep.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
    t = f.MVT_TIME_UTC_mvt
    sec = t.astype("int64") // 10**9 if np.issubdtype(t.dtype, np.datetime64) else t.astype("int64")
    return pd.DataFrame({"MVT_ID_mvt": f.MVT_ID_mvt.astype("int64").to_numpy(), "day": (sec // 86400).to_numpy(),
                         "month": pd.to_datetime(sec, unit="s").dt.month.to_numpy()})


def load_pair(phase, ref, cand):
    if phase == "final":
        a = pd.read_parquet(pred_path(phase, ref), columns=["MVT_ID_mvt", "y", "pred"]).rename(columns={"pred": "p0"})
        b = pd.read_parquet(pred_path(phase, cand), columns=["MVT_ID_mvt", "pred"]).rename(columns={"pred": "p1"})
        d = a.merge(b, on="MVT_ID_mvt")
    else:
        d = pd.read_parquet(ROOT / f"data/{phase}/truth.parquet").rename(columns={"TAXITIME_SEC_mvt": "y"})
        d["MVT_ID_mvt"] = d.MVT_ID_mvt.astype("int64")
        for name, i in (("p0", ref), ("p1", cand)):
            o = pd.read_parquet(pred_path(phase, i), columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"]).rename(columns={"TAXITIME_SEC_mvt": name})
            o["MVT_ID_mvt"] = o.MVT_ID_mvt.astype("int64")
            d = d.merge(o, on="MVT_ID_mvt", how="left")
    d["MVT_ID_mvt"] = d.MVT_ID_mvt.astype("int64")
    d = d.merge(day_month(phase), on="MVT_ID_mvt", how="left")
    if d[["p0", "p1", "day"]].isna().any().any():
        raise SystemExit(f"{phase}: rows without a prediction of both chains or without a day")
    return d[d.month.isin(FOLDS[phase])]


def month_stats(phase, d, refit_sd):
    y, e0, e1 = d.y.to_numpy(float), (d.p0 - d.y).to_numpy(float) ** 2, (d.p1 - d.y).to_numpy(float) ** 2
    out = {}
    for m in FOLDS[phase]:
        k = d.month.to_numpy() == m
        r = {"rows": int(k.sum()), "rmse_ref": float(np.sqrt(e0[k].mean())), "rmse_cand": float(np.sqrt(e1[k].mean()))}
        rng = np.random.default_rng(1000 * list(FOLDS).index(phase) + m)
        for sub, kk in (("all", k), ("trim", k & (y <= TRIM))):
            g = pd.DataFrame({"day": d.day.to_numpy()[kk], "dse": (e1 - e0)[kk]}).groupby("day").dse.agg(["sum", "size"])
            s, n = g["sum"].to_numpy(), g["size"].to_numpy(float)
            idx = rng.integers(0, len(g), (B, len(g)))
            r[sub] = (float(s.sum() / n.sum()), float(np.hypot(np.std(s[idx].sum(1) / n[idx].sum(1)), refit_sd)))
        out[m] = r
    return out


def dl_tau(x, se):
    x, se = np.asarray(x, float), np.maximum(np.asarray(se, float), 1e-9)
    if len(x) < 2:
        return 0.0
    w = 1 / se ** 2
    mu = (w * x).sum() / w.sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    return float(np.sqrt(max(0.0, ((w * (x - mu) ** 2).sum() - (len(x) - 1)) / c))) if c > 0 else 0.0


def estimate(x, se):
    """-> E (MSE), its SD, tau, mu, and the shrunk theta of every test month."""
    tau = dl_tau(list(x.values()), [se[m] for m in x])
    v = {m: 1 / (se[m] ** 2 + tau ** 2) for m in x}
    mu = sum(v[m] * x[m] for m in x) / sum(v.values())
    lam = {T: tau ** 2 / (tau ** 2 + se[T] ** 2) if tau > 0 else 0.0 for T in SHARE}
    theta = {T: lam[T] * x[T] + (1 - lam[T]) * mu for T in SHARE}
    a = {m: 0.0 for m in x}
    for T in SHARE:
        a[T] += SHARE[T] * lam[T]
        for m in x:
            a[m] += SHARE[T] * (1 - lam[T]) * v[m] / sum(v.values())
    var = tau ** 2 * sum((a.get(m, 0.0) - SHARE.get(m, 0.0)) ** 2 for m in range(1, 13)) + sum(a[m] ** 2 * se[m] ** 2 for m in x)
    return sum(SHARE[T] * theta[T] for T in SHARE), float(np.sqrt(var)), tau, mu, theta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cand")
    ap.add_argument("ref")
    ap.add_argument("--phases", nargs="+", default=list(FOLDS))
    ap.add_argument("--refit_sd", type=float, default=0.0)
    a = ap.parse_args()
    stats = {}
    for ph in a.phases:
        stats.update(month_stats(ph, load_pair(ph, a.ref, a.cand), a.refit_sd))
    print(f"{a.cand} against {a.ref}: change per held-out month (negative = {a.cand} better)")
    for m, r in sorted(stats.items()):
        lo, hi = r["all"][0] - 1.96 * r["all"][1], r["all"][0] + 1.96 * r["all"][1]
        print(f"  month {m:2d}: {r['rows']:6d} rows  RMSE {r['rmse_ref']:8.2f} -> {r['rmse_cand']:8.2f}  dMSE {r['all'][0]:+10.1f}  [{lo:+.1f}, {hi:+.1f}]")
    need = [T for T in SHARE if T not in stats]
    if need:
        print(f"months {need} of the test set are not covered by --phases: no weighted estimate")
        return
    for sub in ("all", "trim"):
        x, se = {m: r[sub][0] for m, r in stats.items()}, {m: r[sub][1] for m, r in stats.items()}
        E, sd, tau, mu, theta = estimate(x, se)
        print(f"rows {sub:4s}: E = {E:+.1f} MSE  (95 % interval [{E - 1.96 * sd:+.1f}, {E + 1.96 * sd:+.1f}], about {E / 535:+.3f} RMSE on the test months)  tau {tau:.1f}  mu {mu:+.1f}")
        print("   test months, shrunk dMSE: " + "  ".join(f"{T}: {theta[T]:+.1f}" for T in SHARE))


if __name__ == "__main__":
    main()
