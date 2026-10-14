# SPDX-License-Identifier: GPL-3.0-or-later
"""Leak-safe target encodings.

fit rows: encoded out-of-fold (5 folds grouped by UTC day), so a training row never sees its own label.
apply rows (validation / test): encoded from all fit rows.
Smoothed mean towards the airport-level prior: (sum + m*prior) / (n + m).
"""
import numpy as np
import pandas as pd

KEYS = {
    "stand_rwy": ["apt_stand", "apt_rwy"],
    "stand": ["apt_stand"],
    "rwy": ["apt_rwy"],
    "airline_apt": ["apt", "airline"],
    "oper_apt": ["apt", "AIRCRAFT_OPERATOR_flt"],
    "type_apt": ["apt", "AIRCRAFT_TYPE_mvt"],
    "ades_apt": ["apt", "ADES_mvt"],
    "stand_rwy_hour": ["apt_stand", "apt_rwy", "local_hour"],
    "airline_stand": ["apt_stand", "airline"],
}


def _key(df, cols):
    k = None
    for c in cols:
        s = df[c]
        v = s.astype(object).where(s.notna(), "NA").map(str)
        k = v if k is None else k + "|" + v
    return pd.Series(pd.factorize(k)[0], index=df.index)


def _encode(key_fit, t_fit, prior_fit, key_apply, prior_apply, m):
    g = pd.DataFrame({"k": key_fit.to_numpy(), "t": t_fit, "one": 1.0}).groupby("k").agg(s=("t", "sum"), n=("one", "sum"))
    s = key_apply.map(g["s"]).fillna(0).to_numpy()
    n = key_apply.map(g["n"]).fillna(0).to_numpy()
    return (s + m * prior_apply) / (n + m), n


def add_te(df, fit_mask, apply_mask, targets, keys=KEYS, m=30, n_inner=5, seed=0):
    """targets: dict name -> np.array aligned with df (NaN = ignore row). Returns dict of new columns (full length)."""
    out = {}
    fit_idx = np.flatnonzero(fit_mask)
    app_idx = np.flatnonzero(apply_mask)
    day = (df["MVT_TIME_UTC_mvt"].to_numpy() // 86400)
    rng = np.random.default_rng(seed)
    ud = np.unique(day[fit_idx])
    fold_of_day = dict(zip(ud, rng.integers(0, n_inner, len(ud))))
    inner = np.array([fold_of_day[d] for d in day[fit_idx]])
    apt = df["apt"].astype(str).to_numpy()
    for kname, cols in keys.items():
        key = _key(df, cols)
        for tname, t in targets.items():
            col = np.full(len(df), np.nan)
            cnt = np.full(len(df), np.nan)
            ok = ~np.isnan(t)
            # airport prior from fit rows
            pri = pd.Series(t[fit_idx][ok[fit_idx]]).groupby(apt[fit_idx][ok[fit_idx]]).mean()
            prior = pd.Series(apt).map(pri).fillna(np.nanmean(t[fit_idx])).to_numpy()
            for f in range(n_inner):
                tr = fit_idx[(inner != f) & ok[fit_idx]]
                ap = fit_idx[inner == f]
                col[ap], cnt[ap] = _encode(key.iloc[tr], t[tr], prior[tr], key.iloc[ap], prior[ap], m)
            tr = fit_idx[ok[fit_idx]]
            col[app_idx], cnt[app_idx] = _encode(key.iloc[tr], t[tr], prior[tr], key.iloc[app_idx], prior[app_idx], m)
            out[f"te_{kname}_{tname}"] = col
        out[f"te_{kname}_n"] = cnt
    return out
