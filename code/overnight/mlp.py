# SPDX-License-Identifier: GPL-3.0-or-later
"""Supervised MLP member for ensemble diversity (no DAE pretraining).

Inputs are the DAE matrix (dae.build_matrix): RankGauss numerics + missing indicators, categorical embeddings.
Body: `layers` residual blocks (LayerNorm -> Linear -> SiLU -> Dropout, skip connection) of width `hidden`.
Target: the clipped residual on the AOBT proxy (same as the CatBoost members and dae.py), fit on non-imputed rows.
Per fold the best epoch on the validation fold is kept; the final refit on all 2025 runs the mean best epoch count.
Output is reg-only like dae.py: OOF outputs/overnight/oof/<id>.parquet, test outputs/overnight/test/<id>_raw.parquet,
so blend.py / final_blend.py mix it with p_imp from --p_from.

  python mlp.py --id mlp_v1_fs3 --fs3 1 [--extra ext_metar,...,adsb2fullrel] [--seeds 2] --final 1
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dae import build_matrix
from ocommon import FOLDS, OUTD, REP, report, rmse, save_result, TORCH_DEV

DEV = TORCH_DEV
SCALE = 300.0


class Block(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.norm, self.lin, self.drop = nn.LayerNorm(d), nn.Linear(d, d), nn.Dropout(drop)

    def forward(self, h):
        return h + self.drop(F.silu(self.lin(self.norm(h))))


class MLP(nn.Module):
    def __init__(self, n_num, cards, hidden=1024, layers=4, drop=0.15, emb=16):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(k, min(emb, max(4, int(np.ceil(np.log2(k)) * 2)))) for k in cards])
        d_in = n_num + sum(e.embedding_dim for e in self.embs)
        self.inp = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([Block(hidden, drop) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, xn, xc):
        h = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)], 1)
        h = F.silu(self.inp(h))
        for b in self.blocks:
            h = b(h)
        return self.out(h).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=5000)
    ap.add_argument("--seeds", type=int, default=1, help="average this many seeds per fold (and in the final refit)")
    ap.add_argument("--anchor", default="aobt", choices=["aobt", "lobt"], help="residual anchor: NM AOBT gap, or MVT - LOBT (mix.py lobt_base)")
    ap.add_argument("--seed0", type=int, default=0, help="first seed (a twin run with other seeds)")
    ap.add_argument("--fs3", type=int, default=1)
    ap.add_argument("--extra", default="", help="comma list of extra feature files features_dep_<name>.parquet")
    ap.add_argument("--final", type=int, default=1)
    ap.add_argument("--snap_ep", type=int, default=0, help="refit only: train the full --epochs OneCycle and snapshot after this epoch "
                    "(the source run's mean best epoch = the model its OOF scored; pipeline_bug_audit #3). Writes test/<id>_raw.parquet")
    ap.add_argument("--refit_only", type=int, default=0, help="1 = skip the CV folds (their OOF and result json stay as recorded): refit on all "
                    "2025 with the recorded results/<id>.json final_epochs and write test/<id>_raw.parquet (final phase: new test rows)")
    a = ap.parse_args()
    run = f"mlp-{a.id}"
    folds = list(FOLDS.items())
    total = (len(folds) + (1 if a.final else 0)) * a.seeds * a.epochs + 1
    t0 = time.time()
    report(run, 0, total, "building matrix", purpose="Does a supervised residual MLP add ensemble diversity next to the CatBoost members?")
    df, Xnum, Xcat, cards = build_matrix(fs3=bool(a.fs3), extra=tuple(filter(None, a.extra.split(","))))
    xn_all = torch.tensor(Xnum, device=DEV)
    xc_all = torch.tensor(Xcat, device=DEV)
    del Xnum, Xcat
    y = df["TAXITIME_SEC_mvt"].to_numpy(float)
    imp = (np.abs(df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]) <= 30).to_numpy()
    base = df["d_aobt"].astype(float).clip(0, 3600).fillna(np.nanmedian(df["d_aobt"])).to_numpy()
    if a.anchor == "lobt":  # same anchor as the LOBT CatBoost member (mix_v33): planned rows MVT - LOBT, else the AOBT gap
        lb = df["d_lobt"].astype(float).to_numpy(); base = np.where(np.isnan(lb), base, lb)
    tg = torch.tensor(np.clip(y - base, -a.clip, a.clip) / SCALE, device=DEV, dtype=torch.float32)
    trm = (df["split"] == "train").to_numpy(); month = df["month"].to_numpy()
    res = {"id": a.id, "model": "mlp", "args": vars(a), "folds": {}}
    step = 0

    def predict(model, idx):
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(idx), 32768):
                j = torch.tensor(idx[i:i + 32768], device=DEV)
                out.append(model(xn_all[j], xc_all[j]).float() * SCALE)
        return torch.cat(out).cpu().numpy()

    def fit(tr, va, epochs, seed, label, snap=0):
        """Train one seed; with `va` return (per-epoch val predictions, best epoch), else the last-epoch model
        (or, with `snap`, the model as it was after epoch `snap` of the full schedule)."""
        nonlocal step
        torch.manual_seed(seed)
        model = MLP(xn_all.shape[1], cards, a.hidden, a.layers, a.drop).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * (len(tr) // a.bs + 1))
        tr_t = torch.tensor(tr, device=DEV)
        best, best_ep, best_pv = np.inf, 0, None
        for ep in range(epochs):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
            for i in range(0, len(perm), a.bs):
                j = perm[i:i + a.bs]
                loss = F.mse_loss(model(xn_all[j], xc_all[j]), tg[j])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            step += 1
            if va is None:
                report(run, step, total, f"{label} epoch {ep + 1}/{epochs}")
                if snap and ep + 1 == snap:
                    return model, snap, None
                continue
            pv = predict(model, va) + base[va]
            s = rmse(pv, y[va])
            report(run, step, total, f"{label} epoch {ep + 1}: val rmse {s:.2f} clean {rmse(pv[y[va] <= 7200], y[va][y[va] <= 7200]):.2f}")
            if s < best:
                best, best_ep, best_pv = s, ep + 1, pv
        return model, best_ep, best_pv

    oof, best_eps = [], []
    if a.refit_only:
        n_ep = int(json.loads((REP / "results" / f"{a.id}.json").read_text())["final_epochs"])
        tr = np.flatnonzero(trm & ~imp)
        te = np.flatnonzero((df["split"] == "rank").to_numpy())
        pts = []
        for s in range(a.seeds):
            m, _, _ = fit(tr, None, n_ep, 100 + a.seed0 + s, f"refit seed {s}")
            pts.append(predict(m, te) + base[te])
        pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy()[te], "reg": np.mean(pts, 0)}).to_parquet(
            OUTD / "test" / f"{a.id}_raw.parquet", index=False)
        report(run, total, total, f"refit ({n_ep} epochs, {a.seeds} seeds) written", finished=True)
        return
    if a.snap_ep:
        n_ep = a.snap_ep
        tr = np.flatnonzero(trm & ~imp)
        te = np.flatnonzero((df["split"] == "rank").to_numpy())
        pts = []
        for s in range(a.seeds):
            m, _, _ = fit(tr, None, a.epochs, 100 + a.seed0 + s, f"snap seed {s}", snap=n_ep)
            pts.append(predict(m, te) + base[te])
        pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy()[te], "reg": np.mean(pts, 0)}).to_parquet(
            OUTD / "test" / f"{a.id}_raw.parquet", index=False)
        report(run, total, total, f"snapshot refit at epoch {n_ep}/{a.epochs} written", finished=True)
        return
    for fname, fd in folds:
        tr = np.flatnonzero(trm & np.isin(month, fd["train"]) & ~imp)
        va = np.flatnonzero(trm & np.isin(month, fd["val"]))
        pvs = []
        for s in range(a.seeds):
            _, be, pv = fit(tr, va, a.epochs, a.seed0 + s, f"{fname} seed {s}")
            pvs.append(pv); best_eps.append(be)
        pv = np.mean(pvs, 0); yv = y[va]
        res["folds"][fname] = {"reg_rmse": rmse(pv, yv), "reg_clean_rmse": rmse(pv[yv <= 7200], yv[yv <= 7200]), "best_epochs": best_eps[-a.seeds:]}
        oof.append(pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy()[va], "fold": fname, "y": yv, "reg": pv}))
    o = pd.concat(oof)
    o.to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    res["cv_reg_rmse"] = rmse(o.reg, o.y); res["cv_reg_clean_rmse"] = rmse(o.reg[o.y <= 7200], o.y[o.y <= 7200])
    if a.final:
        # OneCycle ties the schedule to the epoch count, so refit with the mean best epoch count (more data, same shape)
        n_ep = max(2, int(round(np.mean(best_eps))))
        res["final_epochs"] = n_ep
        tr = np.flatnonzero(trm & ~imp)
        te = np.flatnonzero((df["split"] == "rank").to_numpy())
        pts = []
        for s in range(a.seeds):
            m, _, _ = fit(tr, None, n_ep, 100 + a.seed0 + s, f"final seed {s}")
            pts.append(predict(m, te) + base[te])
        pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy()[te], "reg": np.mean(pts, 0)}).to_parquet(
            OUTD / "test" / f"{a.id}_raw.parquet", index=False)
    res["minutes"] = round((time.time() - t0) / 60, 2)
    save_result(res)
    report(run, total, total, f"MLP reg CV {res['cv_reg_rmse']:.2f} clean {res['cv_reg_clean_rmse']:.2f} ({res['minutes']} min)", finished=True)


if __name__ == "__main__":
    main()
