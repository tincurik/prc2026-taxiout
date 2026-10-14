# SPDX-License-Identifier: GPL-3.0-or-later
"""Jahrer-style denoising autoencoder (Porto Seguro 1st place recipe, adapted) + supervised MLP head.

Stage 1 (unsupervised, all train+ranking departures — no labels used):
  numerics -> RankGauss (QuantileTransformer to normal) + missing indicators
  categoricals -> index (rare levels -> 'other'), embedded
  swap noise: each cell replaced by the same column's value from a random row with prob `swap`
  body: 3 x `hidden` ReLU layers; heads reconstruct numerics (MSE) and categoricals (CE)
Stage 2 (supervised, per CV fold): frozen DAE, concat of all hidden layers -> MLP -> non-imputed target
  (clipped residual on the AOBT proxy); OOF saved so it can be blended into the mixture
  pred = p_imp * d_sched + (1 - p_imp) * reg.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer

from bench import CATS, NON_FEATURES
from ocommon import FOLDS, OUTD, PROC, report, rmse, save_result, TORCH_DEV

DEV = TORCH_DEV


META = ["MVT_ID_mvt", "apt", "split", "month", "TAXITIME_SEC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "d_aobt", "d_lobt"]


def build_matrix(max_levels=2000, min_count=20, fs3=False, extra=()):
    """RankGauss numerics + missing indicators (Xnum, float32) and categorical codes (Xcat) for every departure.

    Memory-lean (fits a 12.7 GB Colab VM with the ADS-B extras): each feature file is read on its own straight into one
    preallocated float32 matrix, RankGauss is applied in row blocks into the output buffer, and the returned frame
    holds only the META columns the callers use. Same numbers as the old all-in-pandas version."""
    base = pl.read_parquet(PROC / "features_dep.parquet")
    ids = base.select("MVT_ID_mvt")
    blk = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt"]).filter(
        pl.col("PHASE_mvt") == "DEP").drop("PHASE_mvt")
    sources = [base] + [PROC / f"features_dep_{n}.parquet" for n in (["fs3"] if fs3 else []) + list(extra)]
    seen, plan = set(), []  # (source, its numeric columns); first file wins a duplicated name, like the old left joins
    for src in sources:
        cols = list(src.columns if isinstance(src, pl.DataFrame) else pl.read_parquet_schema(src))
        plan.append((src, [c for c in cols if c not in seen and c not in NON_FEATURES and c not in CATS and c != "BLOCK_TIME_UTC_mvt"]))
        seen.update(cols)
    nums = [c for _, cs in plan for c in cs]
    n = len(ids)
    Xn = np.empty((n, len(nums)), np.float32)
    meta, cat_s, k = base.select([c for c in META if c in base.columns]), {}, 0
    for src, cs in plan:
        t = src if isinstance(src, pl.DataFrame) else ids.join(pl.read_parquet(src), on="MVT_ID_mvt", how="left")
        for c in cs:
            Xn[:, k] = t[c].cast(pl.Float32).to_numpy(); k += 1
        meta = meta.with_columns([t[c] for c in META if c in t.columns and c not in meta.columns])
        cat_s.update({c: t[c] for c in CATS if c in t.columns and c not in cat_s})
        del t
    del base, plan, sources, src  # the base frame is ~2 GB with the extras
    meta = meta.join(blk, on="MVT_ID_mvt", how="left").to_pandas()
    miss_cols = [i for i in range(Xn.shape[1]) if np.isnan(Xn[:, i]).any()]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=0).fit(Xn)
    p = Xn.shape[1]
    Xnum = np.empty((n, p + len(miss_cols)), np.float32)
    for i in range(0, n, 200000):
        Xnum[i:i + 200000, :p] = np.nan_to_num(qt.transform(Xn[i:i + 200000]), nan=0.0)
    Xnum[:, p:] = np.isnan(Xn[:, miss_cols])
    del Xn
    cards, Xc = [], []
    for c in CATS:
        s = cat_s[c].to_pandas().astype(str) if c in cat_s else pd.Series(["nan"] * n)
        vc = s.value_counts()
        keep = vc[vc >= min_count].index[:max_levels]
        mp = {k: i + 1 for i, k in enumerate(keep)}
        Xc.append(s.map(mp).fillna(0).astype(np.int64).to_numpy())
        cards.append(len(keep) + 1)
    Xcat = np.stack(Xc, 1)
    build_matrix.cols = (nums, miss_cols)  # Xnum[:, :len(nums)] = nums in order; Xnum[:, len(nums) + j] = missing flag of nums[miss_cols[j]]
    return meta, Xnum, Xcat, cards


class DAE(nn.Module):
    def __init__(self, n_num, cards, hidden=1024, emb=16, layers=3):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(k, min(emb, max(4, int(np.ceil(np.log2(k)) * 2)))) for k in cards])
        d_in = n_num + sum(e.embedding_dim for e in self.embs)
        self.body = nn.ModuleList()
        d = d_in
        for _ in range(layers):
            self.body.append(nn.Linear(d, hidden)); d = hidden
        self.out_num = nn.Linear(hidden, n_num)
        self.out_cat = nn.ModuleList([nn.Linear(hidden, k) for k in cards])

    def encode(self, xn, xc):
        h = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)], 1)
        hs = []
        for lin in self.body:
            h = F.relu(lin(h)); hs.append(h)
        return hs

    def forward(self, xn, xc):
        h = self.encode(xn, xc)[-1]
        return self.out_num(h), [o(h) for o in self.out_cat]


def swap_noise(x, p):
    n = x.shape[0]
    mask = torch.rand(x.shape, device=x.device) < p
    perm = torch.randint(0, n, x.shape, device=x.device)
    swapped = torch.gather(x, 0, perm)
    return torch.where(mask, swapped, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="dae_v1")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--swap", type=float, default=0.15)
    ap.add_argument("--dae_epochs", type=int, default=12)
    ap.add_argument("--head_epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=1800)
    ap.add_argument("--anchor", default="aobt", choices=["aobt", "lobt"], help="residual anchor: NM AOBT gap, or MVT - LOBT (as mlp.py)")
    ap.add_argument("--fs3", type=int, default=0)
    ap.add_argument("--extra", default="", help="comma list of extra feature files features_dep_<name>.parquet")
    ap.add_argument("--load", default="", help="reuse a trained DAE (<id>_dae.pt, same --hidden/--fs3) and only fit heads")
    ap.add_argument("--final", type=int, default=0, help="also fit a head on all 2025 and write outputs/overnight/test/<id>_raw.parquet")
    ap.add_argument("--refit_only", type=int, default=0, help="1 (with --final 1) = skip the CV folds, whose OOF and result json stay as recorded; "
                    "pretrain the DAE on the matrix of this EC_PHASE and fit only the final head (final-phase replay)")
    a = ap.parse_args()
    run = f"dae-{a.id}"
    total = a.dae_epochs + (3 if a.final else 2) * a.head_epochs + 1
    t0 = time.time()
    report(run, 0, total, "building matrix", purpose="Does a Jahrer swap-noise DAE representation add signal beyond GBDT?")
    df, Xnum, Xcat, cards = build_matrix(fs3=bool(a.fs3), extra=tuple(filter(None, a.extra.split(","))))
    xn_all = torch.tensor(Xnum, device=DEV)
    xc_all = torch.tensor(Xcat, device=DEV)
    N = len(df)
    model = DAE(Xnum.shape[1], cards, hidden=a.hidden).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.dae_epochs * (N // a.bs + 1))
    if a.load:
        model.load_state_dict(torch.load(OUTD / f"{a.load}_dae.pt"))
    for ep in range(0 if a.load else a.dae_epochs):
        model.train(); perm = torch.randperm(N, device=DEV); tot = 0.0
        for i in range(0, N, a.bs):
            idx = perm[i:i + a.bs]
            xn, xc = xn_all[idx], xc_all[idx]
            # swap noise draws donors from the whole batch (batch is a random sample of all rows)
            xn_n, xc_n = swap_noise(xn, a.swap), swap_noise(xc, a.swap)
            with torch.autocast(DEV, dtype=torch.bfloat16):
                rn, rc = model(xn_n, xc_n)
                loss = F.mse_loss(rn.float(), xn) + sum(F.cross_entropy(r.float(), xc[:, j]) for j, r in enumerate(rc)) / len(rc)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        report(run, ep + 1, total, f"DAE epoch {ep+1}/{a.dae_epochs} loss {tot/N:.4f}")
    if not a.load:
        torch.save(model.state_dict(), OUTD / f"{a.id}_dae.pt")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    y = df["TAXITIME_SEC_mvt"].to_numpy(float)
    imp = (np.abs(df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]) <= 30).to_numpy()
    base = df["d_aobt"].astype(float).clip(0, 3600).fillna(np.nanmedian(df["d_aobt"])).to_numpy()
    if a.anchor == "lobt":  # same anchor as mix_v33/v37 and mlp_v4: planned rows MVT - LOBT, else the AOBT gap
        lb = df["d_lobt"].astype(float).to_numpy(); base = np.where(np.isnan(lb), base, lb)
    tgt = np.clip(y - base, -a.clip, a.clip)
    trm = (df["split"] == "train").to_numpy(); month = df["month"].to_numpy()
    res = {"id": a.id, "model": "dae_mlp", "args": vars(a), "folds": {}}
    oof = []
    step = a.dae_epochs
    splits = [(fname, np.flatnonzero(trm & np.isin(month, fd["train"]) & ~imp), np.flatnonzero(trm & np.isin(month, fd["val"])))
              for fname, fd in FOLDS.items()]
    if a.final:
        splits.append(("final", np.flatnonzero(trm & ~imp), np.flatnonzero((df["split"] == "rank").to_numpy())))
    if a.refit_only:
        assert a.final, "--refit_only needs --final 1"
        splits = splits[-1:]
    for fname, tr, va in splits:
        d_rep = 3 * a.hidden
        head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_rep, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)).to(DEV)
        hopt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        hs = torch.optim.lr_scheduler.OneCycleLR(hopt, max_lr=1e-3, total_steps=a.head_epochs * (len(tr) // a.bs + 1))
        tr_t = torch.tensor(tr, device=DEV); tg = torch.tensor(tgt, device=DEV, dtype=torch.float32)
        scale = 300.0

        def predict(idx):
            out = []
            with torch.no_grad():
                for i in range(0, len(idx), 16384):
                    j = torch.tensor(idx[i:i + 16384], device=DEV)
                    with torch.autocast(DEV, dtype=torch.bfloat16):
                        rep = torch.cat(model.encode(xn_all[j], xc_all[j]), 1)
                        out.append(head(rep).float().squeeze(1) * scale)
            return torch.cat(out).cpu().numpy()

        for ep in range(a.head_epochs):
            head.train(); perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
            for i in range(0, len(perm), a.bs):
                j = perm[i:i + a.bs]
                with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
                    rep = torch.cat(model.encode(xn_all[j], xc_all[j]), 1)
                with torch.autocast(DEV, dtype=torch.bfloat16):
                    out = head(rep.float()).float().squeeze(1)
                loss = F.mse_loss(out, tg[j] / scale)
                hopt.zero_grad(set_to_none=True); loss.backward(); hopt.step(); hs.step()
            head.eval()
            step += 1
            if fname == "final":
                report(run, step, total, f"final head epoch {ep+1}")
                continue
            pv = predict(va) + base[va]
            report(run, step, total, f"{fname} head epoch {ep+1}: val(all rows, no mixture) rmse {rmse(pv, y[va]):.2f} clean {rmse(pv[y[va]<=7200], y[va][y[va]<=7200]):.2f}")
        pv = predict(va) + base[va]
        if fname == "final":
            pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy()[va], "reg": pv}).to_parquet(OUTD / "test" / f"{a.id}_raw.parquet", index=False)
            continue
        yv = y[va]
        res["folds"][fname] = {"reg_rmse": rmse(pv, yv), "reg_clean_rmse": rmse(pv[yv <= 7200], yv[yv <= 7200])}
        oof.append(pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy()[va], "fold": fname, "y": yv, "reg": pv}))
    if a.refit_only:
        report(run, total, total, "final head refit done", finished=True)
        return
    o = pd.concat(oof)
    o.to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    res["cv_reg_rmse"] = rmse(o.reg, o.y); res["cv_reg_clean_rmse"] = rmse(o.reg[o.y <= 7200], o.y[o.y <= 7200])
    res["minutes"] = round((time.time() - t0) / 60, 2)
    save_result(res)
    report(run, total, total, f"DAE-MLP reg CV {res['cv_reg_rmse']:.2f} clean {res['cv_reg_clean_rmse']:.2f}", finished=True)


if __name__ == "__main__":
    main()
