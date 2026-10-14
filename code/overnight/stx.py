# SPDX-License-Identifier: GPL-3.0-or-later
"""Stream transformer member (member B of the chain): every movement of an airport, departures and arrivals, is a
token in time order, and a transformer over chunks of `chunk` tokens lets each departure see the clocks, plans and features of
the flights around it (the shared sub-hour level a per-flight member cannot see). Attention carries a learned per-head
time-decay bias (-slope_h * |dt| in hours), so heads range from "the next few minutes" to "the whole afternoon".
  departure tokens  the mlp.py input matrix (dae.build_matrix: RankGauss numerics + missing flags + categorical codes; never BLOCK)
  arrival tokens    taxi-in, delay at in-block, landing vs schedule, time of day, runway / stand / type / operator
                    (all visible in the test files)
Target and contract as mlp.py: clip(y - base, +-clip) / 300 with base = the NM AOBT gap (or the LOBT anchor), loss on
non-imputed departures of the training months; per fold the best epoch on the validation month is kept (its predictions are
the OOF), the final refit on all 2025 runs the mean best epoch count. Output reg-only: outputs/overnight/oof/<id>.parquet
(MVT_ID_mvt, fold, y, reg) and outputs/overnight/test/<id>_raw.parquet (MVT_ID_mvt, reg).
Inference uses chunks with stride chunk/2 and keeps each token's prediction from the chunk where it sits in the central half.
--ctx 0 is the same network with attention to itself only: the no-context ablation (a per-flight MLP of the same shape).
stx_next options (all off = stx_v2):
  --tdir 1    direction-aware time bias: separate per-head decay slopes for flights that took off before (ahead in the queue)
              and after the query (stx_v2's -slope*|dt| cannot tell them apart; the minute of day is only one RankGauss input,
              which dot-product attention cannot turn into a signed offset)
  --rope 1    rotary embedding of the take-off time (minutes) on half of each head's q/k dims, periods 2 min .. 24 h, so
              attention can match "about N minutes before/after me" by content
  --mtm P     masked-token auxiliary: per training chunk a share P of tokens is replaced by a learned mask vector and the
              network reconstructs its numeric inputs (departure RankGauss vector / arrival vector) from the stream (weight
              --mtm_w); the training streams then also hold the label-free months (CV: the validation month; refit: the
              'rank' split), i.e. transductive self-supervision on the test streams; no label of those months is used
  stx_v3 options (all off = stx_tm):
  --adrop P   ADS-B modality dropout: per training batch a share P of the TRACKED departure tokens (a2_match == 1 and a2_ok == 1) gets the
              a2_* block (values + missing flags) overwritten with the untracked encoding (per-column median over rows with no match); y loss kept
  --aux_w W   head rebuilding the true a2_* block of the dropped tokens from the stream (weight W; needs --adrop)
  --adrop_chunk Q  dark-stream dropout: per training chunk with probability Q
              EVERY tracked departure token of the chunk is dropped (rate 1 instead of --adrop), so the model also trains on
              whole dark streams (2026 loses whole EDDF days; token dropout always leaves ~60% of the neighbours lit); needs --adrop
  --sig 1     log-variance head, Gaussian NLL on (target - mu.detach()), weight --head_w; output sig (seconds)
  --tail 1800,3600  one BCE head per threshold on y - base > thr, weight --head_w each; outputs p1800, p3600
  --ema D     EMA of the weights (decay min(D, (1+n)/(10+n))), used for validation and predict
  --knn 50,200  after each fit, cosine top-k of the L2-normalised last hidden state of the query tokens (validation month / test) among
              the labelled training departures: knn_m<k>, knn_s<k> (mean / std of y - base over the non-copy neighbours), knn_d<k> (cosine
              distance to the k-th neighbour), knn_copy<k> (share of SCHED-copy rows among them); computed with the best-epoch weights
  --mask_val final  (CV only; works on any stx incl. stx_tm) second scoring of every validation fold with the ADS-B block hidden (untracked
              encoding) on a seeded share of the tracked validation departures at the final's per-airport loss rates (mix.hide_rates,
              EDDF .289, LIRF .242, rest ~0): OOF columns `hidden`, `reg_masked`; results json masked_* (pooled, hidden
              rows, by airport with bias). The best epoch is still chosen on the plain validation.
  --mask_val hm  (CV only, an analysis option not used by the shipped chain; the row set is data/hide/<phase>_hm.parquet, see ocommon.py) the hidden rows stay dark for the whole fold,
              so the OOF of a hidden row is its 2026-scenario prediction (column `hidden`).
  The new outputs are extra columns of the OOF / test parquets (features_stx.py --cols).
  teacher-student options (all off = stx_v3):
  --kd_w W    privileged self-distillation: the EMA copy (needs --ema) sees the full chunk (no ADS-B dropout, no token mask) and its
              prediction is a second target, weight W, for the ADS-B-dropped tokens (W 1 = target 0.5 y + 0.5 teacher); kd from epoch --kd_start
  --kd_h W    hidden-state distillation on the same tokens: MSE of the layer-normed last hidden states (student vs teacher), weight W
  --kd_unl 1  the same on the dropped tracked tokens of the label-free streams (CV: the validation month; refit: the 'rank' months), no y
  --pl ID     transductive pseudo-labels: the departures of the label-free streams get the target clip(teacher reg - base) with weight --pl_w;
              teacher = the pre-stacker chain ID (no stacker: its CV rows used no label of their own month), read from
              OUTD/oof/ID.parquet (CV folds, column reg) and OUTD/test/ID_parts.parquet (refit, column reg) of the running phase
  neighbour-label head (off = stx_v3 unchanged):
  --nbaux W   two more head2 outputs regress, for each loss-row departure, the leave-one-out sum of clip(y - base, +-nb_clip) / 300 over
              the OTHER loss rows of its airport (nbm) and of its airport + runway (nbr) within +-nb_win s, divided by (n + nb_k)
              (shrunk to 0), weight W each, only where n >= nb_min. Built per fit from that fit's own loss mask, so a CV fold never
              sees its validation month's labels and label-free / 2026 streams get no target. A denoised training signal for the
              shared airport-hour level; outputs nbm<win/60>, nbr<win/60> (seconds) in the stx parquets; --nb_win 900,1800 = one
              airport + runway pair per width (the +-15 min window is the sharpest)
  speed options (all off = same model, different kernels; losses are the same masked means, RNG draws differ):
  --nosync 1  no host sync in a training step: both token encoders on every token + torch.where (no boolean indexing), masked-sum losses,
              the last batch / predict batch padded with empty chunks, loss accumulated as a tensor (1.5x faster on the 5070 Ti)
  --fused 1   fused AdamW kernel
  input encoding (off = as before):
  --ple K     piecewise-linear encoding of the departure numerics: K bins at the normal quantiles, a per-feature linear map to
              --ple_d dims + ReLU before the token MLP (Gorishniy et al. 2022, Q-LR); missing flags and reconstruction targets stay raw
  --compile 1 torch.compile (dynamic=False) of STX.forward (bias build, rotary, attention, MLPs fused); needs --nosync so every batch has the
              same shape (train 32 chunks, predict 64, kd teacher 32); the first epoch pays ~16 s of compile (cached in /tmp/torchinductor_*)
  recurrent trunk (off = attention as before):
  --arch gru  the attention layers are replaced by a 2-layer bidirectional GRU (d/2 per direction, fp32, packed to each chunk's valid
              length) over the same time-ordered chunk, input = token + log1p of the minutes to the previous / next token, then a residual
              FF block; heads, mtm, adrop, kd teacher and every output are unchanged. A second, decorrelated view of the stream: it
              carries backlog token by token instead of pooling it by learned time decay. --tdir/--rope/--rel are ignored; no --compile
  --arch conv the same input through 8 pre-norm residual blocks of non-causal dilated 1D convolutions (k 3, dilation 1..128, +-255 tokens),
              then the FF block (the 1D-CNN member of Optiver Realized Volatility's 1st-place blend); padding is zeroed
              before every conv; runs under the usual autocast
  python stx.py --id stx_v1 --extra ext_metar,ext_prc,ext_osm,ext_opdiev,ext_ord,adsb2fullrel [--ctx 1] [--seeds 1] [--final 1]
  python stx.py --id stx_smoke --smoke 1      tiny CPU run on two airports (code path check, writes nothing under outputs/)
"""
import argparse
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ocommon import FOLDS, OUTD, PROC, REP, hide_ids, report, rmse, save_result

SCALE = 300.0
ADT = torch.bfloat16  # autocast dtype; --amp fp16 (T4 has no bf16) switches it in main()


def emb_dim(k, emb):
    return min(emb, max(4, int(np.ceil(np.log2(max(k, 2)))) * 2))


def rot(x, rope):
    """rotary on the first 2*nf dims of each head: x (B, H, L, hd), rope = (cos, sin) each (B, 1, L, nf)"""
    c, s = rope; nf = c.shape[-1]
    x1, x2, xr = x[..., :nf], x[..., nf:2 * nf], x[..., 2 * nf:]
    c, s = c.to(x.dtype), s.to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c, xr], -1)


class Layer(nn.Module):
    def __init__(self, d, heads, ff, drop):
        super().__init__()
        self.h, self.p = heads, drop
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, bias, rope=None):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = rot(q, rope), rot(k, rope)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias if bias.dtype == q.dtype else bias.to(q.dtype), dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ff(self.n2(x)))


class STX(nn.Module):
    def __init__(self, n_num, cards, n_anum, acards, d=256, layers=4, heads=8, ff=4, drop=0.1, emb=16, tdir=0, rope=0, mtm=0, n_blk=0, nhead=0,
                 ple=0, ple_d=8, n_ple=0, rel=0, arch="attn"):
        super().__init__()
        self.arch = arch
        if arch != "attn":  # --arch gru/conv: the time-decay bias, rotary and relation terms belong to the attention trunk
            tdir, rope, rel = 0, 0, 0
        self.tdir, self.rope, self.mtm, self.rel = tdir, rope, mtm, rel
        self.demb = nn.ModuleList([nn.Embedding(k, emb_dim(k, emb)) for k in cards])
        self.ple, self.n_ple = ple, (n_ple or n_num) if ple else 0
        n_in = n_num
        if ple:  # --ple: bins at the standard-normal quantiles (RankGauss inputs), outer edges at the transform's +-5.2 bounds
            edges = torch.cat([torch.tensor([-5.2]), torch.distributions.Normal(0.0, 1.0).icdf(torch.arange(1, ple) / ple), torch.tensor([5.2])])
            self.register_buffer("ple_lo", edges[:-1].clone())
            self.register_buffer("ple_wid", edges[1:] - edges[:-1])
            bd = 1 / math.sqrt(ple)  # nn.Linear's default init range for fan-in K
            self.ple_w = nn.Parameter(torch.empty(self.n_ple, ple, ple_d).uniform_(-bd, bd))
            self.ple_b = nn.Parameter(torch.empty(self.n_ple, ple_d).uniform_(-bd, bd))
            n_in = n_num - self.n_ple + self.n_ple * ple_d
        self.denc = nn.Sequential(nn.Linear(n_in + sum(e.embedding_dim for e in self.demb), 2 * d), nn.SiLU(), nn.Dropout(drop),
                                  nn.Linear(2 * d, d))
        self.aemb = nn.ModuleList([nn.Embedding(k, emb_dim(k, emb)) for k in acards])
        self.aenc = nn.Sequential(nn.Linear(n_anum + sum(e.embedding_dim for e in self.aemb), d), nn.SiLU(), nn.Linear(d, d))
        self.kind = nn.Embedding(2, d)
        self.layers = nn.ModuleList([Layer(d, heads, ff, drop) for _ in range(layers)] if arch == "attn" else [])
        if arch == "gru":  # input = token + log gaps (min) to the previous / next token; 2-layer bi-GRU (d/2 per direction), residual + FF
            self.gin = nn.Linear(d + 2, d)
            self.rnn = nn.GRU(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
            self.rn1, self.rn2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.rff = nn.Sequential(nn.Linear(d, ff * d), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff * d, d))
            self.rdrop = nn.Dropout(drop)
        if arch == "conv":  # input as gru; 8 pre-norm residual blocks of non-causal dilated convs (k 3, dilation 1..128: +-255 tokens), then FF
            self.gin = nn.Linear(d + 2, d)
            self.cnorm = nn.ModuleList([nn.LayerNorm(d) for _ in range(8)])
            self.convs = nn.ModuleList([nn.Conv1d(d, d, 3, padding=2 ** i, dilation=2 ** i) for i in range(8)])
            self.cpw = nn.ModuleList([nn.Linear(d, d) for _ in range(8)])
            self.rn2 = nn.LayerNorm(d)
            self.rff = nn.Sequential(nn.Linear(d, ff * d), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff * d, d))
            self.rdrop = nn.Dropout(drop)
        self.log_slope = nn.Parameter(torch.log(torch.logspace(math.log10(0.05), math.log10(4.0), heads)))  # per hour
        if tdir:  # slopes for keys that took off after the query (log_slope then covers the ones before)
            self.log_slope_f = nn.Parameter(torch.log(torch.logspace(math.log10(0.05), math.log10(4.0), heads)))
        if rope:
            nf = (d // heads) // 4  # half of the head dims rotate
            self.register_buffer("omega", 2 * math.pi / torch.logspace(math.log10(2.0), math.log10(1440.0), nf))  # rad per minute
        if mtm:
            self.mask_emb = nn.Parameter(torch.zeros(d))
            self.rec_d, self.rec_a = nn.Linear(d, n_num), nn.Linear(d, n_anum)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        if n_blk:
            self.rec_b = nn.Linear(d, n_blk)
        if nhead:  # log-variance + tail logits
            self.head2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, nhead))
        if rel:  # --rel: learned per-head bias on pairs sharing runway / airline / apron (rows 0-2) and per (query kind, key kind) pair (rows 3-6); zero init
            self.rel_w = nn.Parameter(torch.zeros(NREL + 4, heads))
        self.d, self.heads = d, heads

    def dep_embed(self, xn, xc):
        if self.ple:  # t[n, f, k]: share of bin k below x_f (1 when x_f is above the bin, 0 below); z = ReLU(per-feature map of t)
            t = ((xn[:, :self.n_ple].float()[:, :, None] - self.ple_lo) / self.ple_wid).clamp(0, 1)
            z = F.relu(torch.einsum("nfk,fkd->nfd", t, self.ple_w) + self.ple_b).flatten(1)
            xn = torch.cat([z.to(xn.dtype), xn[:, self.n_ple:]], 1)
        return self.denc(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.demb)], 1))

    def arr_embed(self, an, ac):
        return self.aenc(torch.cat([an] + [e(ac[:, i]) for i, e in enumerate(self.aemb)], 1))

    def forward(self, h, t, valid, ctx=True, return_h=False, rk=None):
        """h (B, L, d) token embeddings, t (B, L) int64 seconds, valid (B, L) bool -> (B, L) predictions"""
        B, L = t.shape
        if self.arch == "gru":
            return self.forward_gru(h, t, valid, ctx, return_h)
        if self.arch == "conv":
            return self.forward_conv(h, t, valid, ctx, return_h)
        if self.tdir:  # sd > 0: the key took off before the query
            sd = (t[:, :, None] - t[:, None, :]).float() / 3600.0
            bias = -(self.log_slope.exp()[None, :, None, None] * sd.clamp(min=0)[:, None]
                     + self.log_slope_f.exp()[None, :, None, None] * (-sd).clamp(min=0)[:, None])
        else:
            dt = (t[:, :, None] - t[:, None, :]).abs().float() / 3600.0
            bias = -self.log_slope.exp()[None, :, None, None] * dt[:, None]
        if self.rel:  # rk (B, L, NREL + 1): relation ids (0 = unknown) and the token kind in the last column
            kp = rk[:, :, None, NREL] * 2 + rk[:, None, :, NREL]
            for r in range(NREL + 4):  # pointwise per relation (an einsum came out with the head dim innermost, which SDPA's bias rejects)
                f_ = ((rk[:, :, None, r] == rk[:, None, :, r]) & (rk[:, :, None, r] > 0)) if r < NREL else (kp == r - NREL)
                bias = bias + self.rel_w[r][None, :, None, None] * f_[:, None].float()
        rope = None
        if self.rope:
            ang = ((t - t[:, :1]).float() / 60.0)[:, None, :, None] * self.omega
            rope = (ang.cos(), ang.sin())
        eye = torch.eye(L, dtype=torch.bool, device=t.device)[None, None]
        blocked = (~valid)[:, None, None, :] & ~eye
        if not ctx:
            blocked = ~eye.expand(B, 1, L, L)
        bias = bias.masked_fill(blocked, float("-inf"))
        if torch.is_autocast_enabled():
            bias = bias.to(torch.get_autocast_dtype("cuda" if bias.is_cuda else "cpu"))  # once, not in every layer
        for lay in self.layers:
            h = lay(h, bias, rope)
        out = self.head(h).squeeze(-1)
        return (out, h) if return_h else out


    def forward_gru(self, h, t, valid, ctx, return_h):
        """--arch gru: the same time-ordered chunk read by a bidirectional GRU. Padding is trailing (gather), so the sequences are
        packed to their valid length and the backward direction starts at each chunk's last real token; the GRU runs in fp32."""
        if not ctx:
            raise ValueError("--arch gru has no --ctx 0 ablation")
        B, L = t.shape
        gaps = self.gaps(t, valid)
        dt_ = "cuda" if h.is_cuda else "cpu"
        with torch.autocast(device_type=dt_, enabled=False):
            x = self.gin(torch.cat([h.float(), gaps], -1))
            ln = valid.sum(1).clamp(min=1).cpu()
            pk = nn.utils.rnn.pack_padded_sequence(x, ln, batch_first=True, enforce_sorted=False)
            o, _ = self.rnn(pk)
            o = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=L)[0]
            x = x + self.rdrop(self.rn1(o))
            x = x + self.rdrop(self.rff(self.rn2(x)))
        h = x.to(h.dtype)
        out = self.head(h).squeeze(-1)
        return (out, h) if return_h else out


    def gaps(self, t, valid):
        """log1p of the minutes to the previous / next real token (0 at the chunk edges and on padding, whose t is arbitrary)"""
        tf = t.float()
        gp = torch.cat([tf[:, :1], tf[:, :-1]], 1); gn = torch.cat([tf[:, 1:], tf[:, -1:]], 1)
        gaps = torch.stack([torch.log1p((tf - gp).clamp(min=0) / 60.0), torch.log1p((gn - tf).clamp(min=0) / 60.0)], -1)
        vn = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], 1)
        return gaps * torch.stack([valid, valid & vn], -1)

    def forward_conv(self, h, t, valid, ctx, return_h):
        """--arch conv: the same time-ordered chunk read by a stack of non-causal dilated 1D convolutions (the CNN member of Optiver's
        1st-place blend); padding is zeroed before every conv, so real tokens never read it."""
        if not ctx:
            raise ValueError("--arch conv has no --ctx 0 ablation")
        m = valid[..., None]
        x = self.gin(torch.cat([h, self.gaps(t, valid).to(h.dtype)], -1))
        for nm, cv, pw in zip(self.cnorm, self.convs, self.cpw):
            y = (nm(x) * m).transpose(1, 2)
            x = x + self.rdrop(pw(F.silu(cv(y).transpose(1, 2))))
        x = x + self.rdrop(self.rff(self.rn2(x)))
        h = x * m
        out = self.head(h).squeeze(-1)
        return (out, h) if return_h else out


NREL = 3  # --rel relations: same runway, same airline, same apron (stand letters)


def nb_targets(lm, t, grp, r, win, k=5.0, min_n=3):
    """--nbaux target of every lm row: sum of r over the OTHER lm rows of its group (grp >= 0, -1 = none) with |t - t'| <= win,
    divided by (n + k); ok = n >= min_n. Rows outside lm get 0 / False. lm bool, t int seconds, grp int, r float (NaN = no label)."""
    out, ok = np.zeros(len(r), np.float32), np.zeros(len(r), bool)
    ix = np.flatnonzero(lm & (grp >= 0) & np.isfinite(r))
    if len(ix) == 0:
        return out, ok
    ix = ix[np.lexsort((t[ix], grp[ix]))]
    key = grp[ix].astype(np.int64) * 10 ** 10 + t[ix].astype(np.int64)  # times < 1e10 s, so a window never crosses groups
    lo, hi = np.searchsorted(key, key - win, "left"), np.searchsorted(key, key + win, "right")
    rr = r[ix].astype(np.float64); cs = np.r_[0.0, np.cumsum(rr)]
    n = hi - lo - 1
    out[ix] = ((cs[hi] - cs[lo] - rr) / (n + k)).astype(np.float32); ok[ix] = n >= min_n
    return out, ok


def rel_keys(dep_ids):
    """(n_dep, NREL), (n_arr, NREL) int64 relation ids shared by departures (rows in dep_ids order) and arrivals (arr_tokens order); 0 = unknown"""
    m = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "RUNWAY_mvt", "STAND_mvt"])
    m = m.with_columns(apt=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).cast(pl.Utf8))
    m = m.select(pl.col("MVT_ID_mvt"), pl.col("PHASE_mvt"),
                 rw=pl.concat_str([pl.col("apt"), pl.col("RUNWAY_mvt").cast(pl.Utf8)], separator="|", ignore_nulls=False),
                 al=pl.col("FLIGHT_mvt").cast(pl.Utf8).str.extract(r"^([A-Z]{3})", 1),
                 zn=pl.concat_str([pl.col("apt"), pl.col("STAND_mvt").cast(pl.Utf8).str.extract(r"^([A-Za-z]+)", 1)], separator="|", ignore_nulls=False)).to_pandas()
    d = m[m.PHASE_mvt == "DEP"].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").reindex(dep_ids)
    r = m[m.PHASE_mvt == "ARR"]
    D, A_ = [], []
    for c in ("rw", "al", "zn"):
        codes, _ = pd.factorize(pd.concat([d[c], r[c]], ignore_index=True))  # -1 for null
        codes = (codes + 1).astype(np.int64)
        D.append(codes[:len(d)]); A_.append(codes[len(d):])
    return np.stack(D, 1), np.stack(A_, 1)


def arr_tokens(min_count=20, max_levels=2000):
    a = pl.read_parquet(PROC / "movements.parquet", columns=["PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                        "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "split"]).filter(
        pl.col("PHASE_mvt") == "ARR").to_pandas()
    t = a.MVT_TIME_UTC_mvt.to_numpy(np.int64)
    blk, sch = a.BLOCK_TIME_UTC_mvt.astype(float).to_numpy(), a.SCHED_TIME_UTC_mvt.astype(float).to_numpy()
    ti, dl, ls = blk - t, blk - sch, t - sch
    tod = 2 * np.pi * (t % 86400) / 86400
    num = np.stack([np.clip(ti, 0, 3600) / 600, np.clip(dl, -3600, 10800) / 1800, np.clip(ls, -3600, 10800) / 1800,
                    np.isnan(ti), np.isnan(dl), np.sin(tod), np.cos(tod)], 1)
    num = np.nan_to_num(num).astype(np.float32)
    apt = a.ADES_mvt.astype(str)
    cats, cards = [], []
    for s in (apt, apt + "|" + a.RUNWAY_mvt.astype(str), apt + "|" + a.STAND_mvt.astype(str), a.AIRCRAFT_TYPE_mvt.astype(str),
              a.AIRCRAFT_OPERATOR_flt.astype(str)):
        vc = s.value_counts(); keep = vc[vc >= min_count].index[:max_levels]
        cats.append(s.map({k: i + 1 for i, k in enumerate(keep)}).fillna(0).astype(np.int64).to_numpy()); cards.append(len(keep) + 1)
    month = pd.to_datetime(t, unit="s", utc=True).month.to_numpy()
    return dict(apt=apt.to_numpy(), t=t, split=a.split.astype(str).to_numpy(), month=month, num=num, cat=np.stack(cats, 1), cards=cards)


def smoke_matrix():
    """cheap stand-in for build_matrix on two airports (code-path check only)"""
    num = ["d_aobt", "d_eobt", "d_sched", "d_lobt", "local_hour", "dow", "nb_daobt_pm30", "queue_at_mvt", "dep_prev_30", "dep_next_30",
           "arr_taxiin_pm60", "rwy_gap_prev", "rwy_gap_next"]
    f = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "split", "month", "TAXITIME_SEC_mvt",
                                                                "SCHED_TIME_UTC_mvt"] + num).to_pandas()
    blk = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt"]).filter(
        pl.col("PHASE_mvt") == "DEP").drop("PHASE_mvt").to_pandas()
    f = f.merge(blk, on="MVT_ID_mvt", how="left")
    X = f[num].astype(float); X = ((X - X.mean()) / X.std()).fillna(0).to_numpy(np.float32)
    s = f.apt.astype(str); keep = list(s.value_counts().index)
    Xc = s.map({k: i + 1 for i, k in enumerate(keep)}).fillna(0).astype(np.int64).to_numpy()[:, None]
    meta = f[["MVT_ID_mvt", "apt", "split", "month", "TAXITIME_SEC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "d_aobt", "d_lobt"]]
    return meta, X, Xc, [len(keep) + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--extra", default="", help="comma list of extra feature files features_dep_<name>.parquet (as mlp.py)")
    ap.add_argument("--fs3", type=int, default=1)
    ap.add_argument("--anchor", default="aobt", choices=["aobt", "lobt"])
    ap.add_argument("--ctx", type=int, default=1, help="0 = attention to itself only (no-context ablation)")
    ap.add_argument("--arr", type=int, default=1, help="0 = departure tokens only")
    ap.add_argument("--tdir", type=int, default=0, help="1 = separate decay slopes for earlier / later take-offs")
    ap.add_argument("--rope", type=int, default=0, help="1 = rotary embedding of take-off minutes on half the head dims")
    ap.add_argument("--mtm", type=float, default=0.0, help="masked-token share for the reconstruction auxiliary (0 = off)")
    ap.add_argument("--mtm_w", type=float, default=0.3)
    ap.add_argument("--pre_ep", type=int, default=0, help="stage 1: this many extra leading epochs of masked-token reconstruction only "
                    "(no supervised loss, no validation), then the supervised epochs; the one-cycle schedule spans both")
    ap.add_argument("--pre_mtm", type=float, default=0.3, help="masked share during the --pre_ep stage")
    ap.add_argument("--unl_rank", type=int, default=0, help="1 = CV folds also train the reconstruction on the label-free 'rank' (2026) streams "
                    "(the refit already does)")
    ap.add_argument("--adrop", type=float, default=0.0, help="share of tracked departure tokens whose a2_* ADS-B block is replaced by the "
                    "untracked encoding per training batch (0 = off; needs adsb2fullrel in --extra)")
    ap.add_argument("--aux_w", type=float, default=0.0, help="weight of the head rebuilding the dropped a2_* block (needs --adrop)")
    ap.add_argument("--adrop_chunk", type=float, default=0.0, help="share of training chunks whose tracked departure tokens are ALL dropped "
                    "(a dark stream, as on a 2026 outage day; needs --adrop; 0 = off)")
    ap.add_argument("--sig", type=int, default=0, help="1 = log-variance head (Gaussian NLL on the detached mean's residual), output sig")
    ap.add_argument("--tail", default="", help="comma list of thresholds in seconds: one BCE head per y - base > thr, outputs p<thr>")
    ap.add_argument("--head_w", type=float, default=0.1, help="loss weight of each --sig / --tail head")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights, used for validation and predict (0 = off)")
    ap.add_argument("--knn", default="", help="comma list of k: neighbour residual statistics in the learned space (knn_m<k>, knn_s<k>, "
                    "knn_d<kmax>, knn_copy)")
    ap.add_argument("--mask_val", default="", choices=["", "final", "hm"], help="also score each validation fold with tracks hidden at the final's loss rates; "
                    "hm = the masked mirror's fixed hidden set (ocommon.hide_ids; analysis option, not used by the shipped chain), dark for the whole fold (the 2026 "
                    "scenario: their tokens feed the label-free terms untracked too), so every OOF column of a hidden row is its masked prediction")
    ap.add_argument("--kd_w", type=float, default=0.0, help="weight of the EMA full-view teacher target on ADS-B-dropped tokens (0 = off)")
    ap.add_argument("--kd_unl", type=int, default=0, help="1 = also distil the dropped tracked tokens of the label-free streams")
    ap.add_argument("--kd_start", type=int, default=2, help="first epoch (0-based) with the kd loss")
    ap.add_argument("--kd_h", type=float, default=0.0, help="weight of hidden-state distillation on the same tokens (layer-normed last hidden "
                    "state of the student vs the full-view EMA teacher; needs --kd_w > 0)")
    ap.add_argument("--pl", default="", help="pre-stacker chain id whose reg is the pseudo-label of the label-free streams' departures")
    ap.add_argument("--pl_w", type=float, default=1.0)
    ap.add_argument("--nbaux", type=float, default=0.0, help="weight of each neighbour-label head (airport / runway leave-one-out "
                    "level within +-nb_win; 0 = off)")
    ap.add_argument("--nb_win", default="1800", help="half-width (s) of the --nbaux neighbour window; a comma list ("
                    "900,1800) gives one airport + one runway head per width, outputs nbm<w/60>, nbr<w/60> in list order")
    ap.add_argument("--nb_k", type=float, default=5.0, help="--nbaux shrinkage: the neighbour sum is divided by n + nb_k")
    ap.add_argument("--nb_min", type=int, default=3, help="--nbaux target only where at least this many neighbours")
    ap.add_argument("--nb_clip", type=float, default=1800.0, help="clip (s) of y - base inside the --nbaux sums (giants would dominate them)")
    ap.add_argument("--arch", default="attn", choices=["attn", "gru", "conv"], help="trunk: attn = the time-decay attention layers (default, unchanged); "
                    "gru = a 2-layer bidirectional GRU over the same time-ordered chunk with log gaps to the neighbouring tokens ("
                    "--tdir/--rope/--rel/--layers/--heads are ignored, no --compile); conv = 8 residual blocks of non-causal dilated 1D convolutions "
                    "(k 3, dilation 1..128) over the same chunk, same inputs")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32, help="chunks per batch")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--ff", type=int, default=4, help="feed-forward width multiplier (set by a hyper-parameter search not included here; recorded values in recipes/)")
    ap.add_argument("--ple", type=int, default=0, help="bins of the piecewise-linear encoding of the departure numerics (0 = raw inputs)")
    ap.add_argument("--ple_d", type=int, default=8, help="dims per numeric feature under --ple")
    ap.add_argument("--rel", type=int, default=0, help="1 = relation bias in the attention: learned per-head terms for key/query pairs on the same runway, airline, apron (stand letters) and per (dep/arr, dep/arr) kind pair (0 = unchanged CV)")
    ap.add_argument("--pct", type=float, default=0.1, help="OneCycle warm-up share of the steps (set by the same search)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=5000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--final", type=int, default=1)
    ap.add_argument("--refit_only", type=int, default=0, help="1 = refit on all 2025 with the recorded final_epochs, write test only")
    ap.add_argument("--nosync", type=int, default=0, help="1 = static shapes, no host syncs in a step (all-token embedding + torch.where, "
                    "masked-sum losses, padded last batch, tensor loss accumulation)")
    ap.add_argument("--compile", type=int, default=0, help="1 = torch.compile of the transformer body (2 = max-autotune-no-cudagraphs); needs --nosync")
    ap.add_argument("--fused", type=int, default=0, help="1 = fused AdamW")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16"], help="autocast dtype; fp16 adds a GradScaler (Kaggle T4 has no bf16)")
    a = ap.parse_args()
    global ADT
    ADT = torch.float16 if a.amp == "fp16" else torch.bfloat16
    dev = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    if a.smoke:
        a.chunk, a.bs, a.d, a.layers, a.heads, a.epochs, a.final = 64, 8, 32, 2, 4, 2, 1
    run = f"stx-{a.id}"
    folds = list(FOLDS.items())
    total = (len(folds) + (1 if a.final else 0)) * a.seeds * (a.epochs + a.pre_ep) + 1
    t0 = time.time()
    report(run, 0, total, "building matrix", purpose="Does a transformer over each airport's movement stream (context of the surrounding flights) "
                                                     "beat the same inputs per flight (ctx 0)?")
    if a.smoke:
        df, Xnum, Xcat, cards = smoke_matrix()
    else:
        from dae import build_matrix
        df, Xnum, Xcat, cards = build_matrix(fs3=bool(a.fs3), extra=tuple(filter(None, a.extra.split(","))))
    n_ple = Xnum.shape[1] if a.smoke else len(build_matrix.cols[0])  # --ple encodes the RankGauss columns; the missing flags stay raw
    tails = [int(t) for t in a.tail.split(",") if t]
    knns = [int(k) for k in a.knn.split(",") if k]
    nb_wins = [int(w) for w in str(a.nb_win).split(",") if w]
    nnb = 2 * len(nb_wins) if a.nbaux > 0 else 0
    nb_names = [f"{h}{w // 60}" for w in nb_wins for h in ("nbm", "nbr")] if nnb else []
    nhead = a.sig + len(tails) + nnb
    blk_idx, trk = [], np.zeros(len(df), bool)
    if a.adrop > 0 or a.mask_val:
        if a.smoke:
            blk_idx = [0, 1, 2]; trk = np.random.default_rng(0).random(len(df)) < 0.5; unm = ~trk
        else:
            assert "adsb2fullrel" in a.extra, "--adrop needs adsb2fullrel in --extra"
            import dae
            nums, miss_cols = dae.build_matrix.cols
            blk_idx = [i for i, c in enumerate(nums) if c.startswith("a2_")] + [len(nums) + j for j, ci in enumerate(miss_cols)
                                                                                  if nums[ci].startswith("a2_")]
            ad = pd.read_parquet(PROC / "features_dep_adsb2fullrel.parquet", columns=["MVT_ID_mvt", "a2_match", "a2_ok"]).set_index(
                "MVT_ID_mvt").reindex(df["MVT_ID_mvt"].to_numpy())
            trk = ((ad.a2_match == 1) & (ad.a2_ok == 1)).to_numpy(); unm = ad.a2_match.isna().to_numpy(); del ad
        hrate = {"EDDM": 0.3, "LSZH": 0.3} if a.smoke else __import__("mix").hide_rates()
        print(f"hide rates {hrate}", flush=True)
        print(f"adrop: block {len(blk_idx)} cols, tracked {trk.mean():.3f}, unmatched {unm.mean():.3f}", flush=True)
        unt_vec = torch.tensor(np.median(Xnum[unm][:, blk_idx], 0), dtype=torch.float32, device=dev)
    tq = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
    assert (tq.MVT_ID_mvt.to_numpy() == df.MVT_ID_mvt.to_numpy()).all(), "build_matrix row order differs from features_dep"
    dep_t = tq.MVT_TIME_UTC_mvt.to_numpy(np.int64); del tq
    A = arr_tokens()
    rel_d = rel_a = None
    if a.rel:
        rd, ra = rel_keys(df["MVT_ID_mvt"].to_numpy()); assert len(ra) == len(A["t"]), "arrival row count differs from arr_tokens"
        rel_d, rel_a = torch.tensor(rd, device=dev), torch.tensor(ra, device=dev)
        print(f"rel: runway/airline/apron known on dep {(rd > 0).mean(0).round(3)}, arr {(ra > 0).mean(0).round(3)}", flush=True)
    dep_apt = df["apt"].astype(str).to_numpy(); APTS = sorted(set(dep_apt))
    if a.smoke:
        APTS = ["EDDM", "LSZH"]
    y = df["TAXITIME_SEC_mvt"].to_numpy(float)
    imp = (np.abs(df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]) <= 30).to_numpy()
    base = df["d_aobt"].astype(float).clip(0, 3600).fillna(np.nanmedian(df["d_aobt"])).to_numpy()
    if a.anchor == "lobt":
        lb = df["d_lobt"].astype(float).to_numpy(); base = np.where(np.isnan(lb), base, lb)
    trm = (df["split"] == "train").to_numpy(); month = df["month"].to_numpy(); rank = (df["split"] == "rank").to_numpy()
    a_tr = A["split"] == "train"; a_rank = A["split"] == "rank"
    # device tensors
    xn_all = torch.tensor(Xnum, dtype=torch.float16, device=dev); xc_all = torch.tensor(Xcat, device=dev)
    an_all = torch.tensor(A["num"], device=dev); ac_all = torch.tensor(A["cat"], device=dev)
    del Xnum, Xcat
    tg = torch.tensor(np.nan_to_num(np.clip(y - base, -a.clip, a.clip) / SCALE), device=dev, dtype=torch.float32)
    tailt = torch.tensor(np.stack([(y - base > t) for t in tails], 1).astype(np.float32), device=dev) if tails else None
    trk_all = torch.tensor(trk, device=dev)
    resid = torch.tensor(np.nan_to_num(np.clip(y - base, -a.clip, a.clip)), device=dev, dtype=torch.float32)
    copy_all = torch.tensor(imp, device=dev)
    nb_g = r_nb = None
    if nnb:  # --nbaux groups: airport, airport + departure runway (-1 = runway unknown); targets are built per fit in fit()
        rw = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "RUNWAY_mvt"]).filter(
            pl.col("PHASE_mvt") == "DEP").to_pandas().drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").RUNWAY_mvt.reindex(
            df["MVT_ID_mvt"].to_numpy())
        g_rw = pd.factorize(pd.Series(dep_apt) + "|" + rw.astype(str).to_numpy())[0]
        nb_g = (pd.factorize(dep_apt)[0], np.where(rw.isna().to_numpy(), -1, g_rw))
        r_nb = np.clip(y - base, -a.nb_clip, a.nb_clip) / SCALE
        print(f"nbaux: runway known on {(nb_g[1] >= 0).mean():.3f} of departures", flush=True)
        del rw, g_rw
    assert a.kd_w + a.kd_h == 0 or (a.ema > 0 and a.adrop > 0), "--kd_w / --kd_h need --ema and --adrop"
    assert a.adrop_chunk == 0 or a.adrop > 0, "--adrop_chunk needs --adrop"
    pl_t = pl_ok = None
    if a.pl:  # pseudo-label targets for every row the teacher covers (CV OOF of the val months + refit test rows); used only on unl rows
        if a.smoke:
            tv = pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], "reg": np.nan_to_num(y, nan=900.0) * 0.9 + 60})
        else:
            tv = pd.concat([pd.read_parquet(OUTD / "oof" / f"{a.pl}.parquet", columns=["MVT_ID_mvt", "reg"]),
                            pd.read_parquet(OUTD / "test" / f"{a.pl}_parts.parquet", columns=["MVT_ID_mvt", "reg"])])
            tv = tv.drop_duplicates("MVT_ID_mvt", keep="first")  # an id in both: the OOF (no own-month label) wins
        tr_ = tv.set_index("MVT_ID_mvt").reg.reindex(df["MVT_ID_mvt"].to_numpy()).to_numpy(float)
        pl_ok = torch.tensor(~np.isnan(tr_), device=dev)
        pl_t = torch.tensor(np.nan_to_num(np.clip(tr_ - base, -a.clip, a.clip) / SCALE), device=dev, dtype=torch.float32)
        print(f"pl {a.pl}: teacher on {np.isfinite(tr_).mean():.3f} of rows, val/rank cover "
              f"{np.isfinite(tr_[(df['split'] == 'rank').to_numpy()]).mean():.3f} (rank)", flush=True)
        del tv, tr_
    out_names = ["reg"] + (["sig"] if a.sig else []) + [f"p{t}" for t in tails] + nb_names
    L = a.chunk; S = L // 2
    ar = torch.arange(L, device=dev)

    def streams(dsel, asel):
        """concatenated per-airport time-sorted tokens (kind 0 = departure, 1 = arrival; idx into its matrix) + stream bounds"""
        ks, ix, ts, bounds, n0 = [], [], [], [], 0
        for apt in APTS:
            di = np.flatnonzero(dsel & (dep_apt == apt))
            ai = np.flatnonzero(asel & (A["apt"] == apt)) if a.arr else np.zeros(0, np.int64)
            if len(di) == 0:
                continue
            tt = np.r_[dep_t[di], A["t"][ai]]; o = np.argsort(tt, kind="stable")
            ks.append(np.r_[np.zeros(len(di), np.int64), np.ones(len(ai), np.int64)][o]); ix.append(np.r_[di, ai][o]); ts.append(tt[o])
            bounds.append((n0, n0 + len(tt))); n0 += len(tt)
        tok = tuple(torch.tensor(np.concatenate(v), device=dev) for v in (ks, ix, ts))
        return tok, bounds

    fwd = STX.forward
    if a.compile:
        assert a.arch == "attn", "--compile is for the attention trunk (cuDNN GRU breaks the graph; conv untested)"
        assert a.nosync, "--compile needs --nosync (static shapes)"
        fwd = torch.compile(STX.forward, dynamic=False, mode="max-autotune-no-cudagraphs" if a.compile == 2 else None)
    blk_mask = unt_full = blk_t = None
    if blk_idx:
        blk_t = torch.tensor(blk_idx, device=dev)
        blk_mask = torch.zeros(xn_all.shape[1], dtype=torch.bool, device=dev); blk_mask[blk_t] = True
        unt_full = torch.zeros(xn_all.shape[1], device=dev); unt_full[blk_t] = unt_vec

    def embed_static(model, kind, idx, valid, mask=None, drp=None):
        """--nosync: both token encoders on every token and torch.where picks (no boolean indexing, no host sync, static shapes)"""
        B, Lb = kind.shape
        i0 = torch.where(kind == 0, idx, 0).reshape(-1); i1 = torch.where(kind == 1, idx, 0).reshape(-1)
        xs = xn_all[i0].float()
        if drp is not None:
            xs = torch.where(drp.reshape(-1, 1) & blk_mask[None, :], unt_full[None, :], xs)
        hd = model.dep_embed(xs, xc_all[i0])
        ha = model.arr_embed(an_all[i1], ac_all[i1]).to(hd.dtype)
        h = (torch.where((kind == 0).reshape(-1, 1), hd, ha) * valid.reshape(-1, 1)).view(B, Lb, model.d)
        if mask is not None:
            h = torch.where(mask[:, :, None], model.mask_emb.to(h.dtype), h)
        return h + model.kind(kind).to(h.dtype)

    def embed(model, kind, idx, valid, mask=None, drp=None):
        if a.nosync:
            return embed_static(model, kind, idx, valid, mask, drp)
        B, Lb = kind.shape
        k, i, v = kind.reshape(-1), idx.reshape(-1), valid.reshape(-1)
        ds, as_ = (k == 0) & v, (k == 1) & v
        xs = xn_all[i[ds]].float()
        if drp is not None:  # ADS-B dropout: tracked tokens take the untracked encoding of the a2_* block
            sel = drp.reshape(-1)[ds]
            if sel.any():
                xd = xs[sel]; xd[:, blk_idx] = unt_vec; xs[sel] = xd
        hd = model.dep_embed(xs, xc_all[i[ds]])
        h = torch.zeros(B * Lb, model.d, device=dev, dtype=hd.dtype)
        h[ds] = hd
        if as_.any():
            h[as_] = model.arr_embed(an_all[i[as_]], ac_all[i[as_]]).to(hd.dtype)
        h = h.view(B, Lb, model.d)
        if mask is not None:
            h = torch.where(mask[:, :, None], model.mask_emb.to(h.dtype), h)
        return h + model.kind(kind).to(h.dtype)

    def rk_of(kind, idx):
        """(B, L, NREL + 1) relation ids of the tokens + their kind (None without --rel)"""
        if rel_d is None:
            return None
        r = torch.where((kind == 0)[..., None], rel_d[torch.where(kind == 0, idx, 0)], rel_a[torch.where(kind == 1, idx, 0)])
        return torch.cat([r, kind[..., None]], -1)

    def gather(tok, st, ln):
        pos = st[:, None] + ar[None, :]
        valid = ar[None, :] < ln[:, None]
        pos = torch.where(valid, pos, st[:, None])
        return tok[0][pos], tok[1][pos], tok[2][pos], valid

    def predict(model, tok, bounds, n_out, want_h=False, hide=None):
        """stride-S chunks; each token keeps the prediction from the chunk where it sits in the central half.
        hide = (n_out,) bool tensor of rows whose a2_* block is replaced by the untracked encoding. Returns {name: (n_out,) array} for reg (seconds, without base), sig, p<thr>; want_h: also the (n_out, d) fp16 hidden states (GPU)"""
        ch = []
        for lo_b, hi_b in bounds:
            k = 0
            while True:
                s = lo_b + k * S; e = min(s + L, hi_b)
                ch.append((s, e - s, 0 if k == 0 else S // 2, (e - s) if e >= hi_b else S // 2 + S))
                if e >= hi_b:
                    break
                k += 1
        ch = torch.tensor(ch, device=dev)
        out = torch.full((len(out_names), n_out), float("nan"), device=dev)
        Hb = torch.zeros((n_out, model.d), dtype=torch.float16, device=dev) if want_h else None
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=ADT, enabled=dev == "cuda"):
            for i in range(0, len(ch), a.bs * 2):
                c = ch[i:i + a.bs * 2]
                if a.nosync and len(c) < a.bs * 2:  # static batch shape: pad with empty chunks
                    c = torch.cat([c, c.new_zeros(a.bs * 2 - len(c), 4)])
                kind, idx, t, valid = gather(tok, c[:, 0], c[:, 1])
                dh = (valid & (kind == 0) & hide[torch.where(kind == 0, idx, 0)]) if hide is not None else None
                p, hh = fwd(model, embed(model, kind, idx, valid, None, dh), t, valid, ctx=bool(a.ctx), return_h=True, rk=rk_of(kind, idx))
                vals = [p.float() * SCALE]
                if nhead:
                    e = model.head2(hh).float()
                    if a.sig:
                        vals.append(torch.exp(0.5 * e[..., 0].clamp(-8, 4)) * SCALE)
                    vals += [torch.sigmoid(e[..., a.sig + j]) for j in range(len(tails))]
                    vals += [e[..., a.sig + len(tails) + j] * SCALE for j in range(nnb)]
                keep = valid & (kind == 0) & (ar[None, :] >= c[:, 2:3]) & (ar[None, :] < c[:, 3:4])
                for j, v in enumerate(vals):
                    out[j, idx[keep]] = v[keep]
                if want_h:
                    Hb[idx[keep]] = hh[keep].to(torch.float16)
        res = {n: out[j].cpu().numpy() for j, n in enumerate(out_names)}
        return (res, Hb) if want_h else res

    def knn_feats(model, qry, ktok, kb, krows):
        """neighbour residual statistics of the query departures among the labelled training departures `krows`, in the model's last hidden space"""
        qtok, qb, qrows = qry
        _, Hk = predict(model, ktok, kb, len(y), want_h=True)
        kr = torch.tensor(krows, device=dev)
        keys = torch.cat([F.normalize(Hk[kr[i:i + 200000]].float(), dim=1).half() for i in range(0, len(kr), 200000)])
        del Hk
        kres, kcp = resid[kr], copy_all[kr].float()
        _, Hq = predict(model, qtok, qb, len(y), want_h=True)
        qr = torch.tensor(qrows, device=dev)
        kmax = max(knns); res = {f"{n}{k}": [] for k in knns for n in ("knn_m", "knn_s")}; res[f"knn_d{kmax}"] = []; res["knn_copy"] = []
        for i in range(0, len(qr), 256):
            q = F.normalize(Hq[qr[i:i + 256]].float(), dim=1).half()
            sim, ind = (q @ keys.T).topk(kmax, dim=1)
            r, cp = kres[ind], kcp[ind]
            w = 1.0 - cp
            for k in knns:
                wk, rk = w[:, :k], r[:, :k]
                n_ = wk.sum(1).clamp(min=1)
                m_ = (rk * wk).sum(1) / n_
                res[f"knn_m{k}"].append(m_)
                res[f"knn_s{k}"].append((((rk - m_[:, None]) ** 2 * wk).sum(1) / n_).sqrt())
            res[f"knn_d{kmax}"].append(1.0 - sim[:, -1].float()); res["knn_copy"].append(cp.mean(1))
        del keys, Hq
        return {k: torch.cat(v).cpu().numpy() for k, v in res.items()}

    hid_t = None  # set per CV fold under --mask_val
    fit_box = {}  # per-fit tensors the static loss needs (loss mask, label-free mask)

    def static_loss(model, ema, p, hh, kind, idx, valid, t, mk, dr, pre, ep, xls):
        """--nosync: the fit-loop losses as masked means over every token (same terms as the boolean-indexed ones, no host sync)"""
        unl_t = fit_box["unl_t"]
        isd = valid & (kind == 0)
        i0 = torch.where(kind == 0, idx, 0)
        m = isd & fit_box["lm_all"][i0]
        if pre:
            m = m & False
        if mk is not None:
            m = m & ~mk
        mf = m.float(); nm = mf.sum().clamp(min=1.0)
        loss = (((p - tg[i0]) ** 2) * mf).sum() / nm
        if nhead:
            e = model.head2(hh.float())
            if a.sig:
                lv = e[..., 0].clamp(-8, 4)
                loss = loss + a.head_w * ((0.5 * (lv + (tg[i0] - p.detach()) ** 2 / lv.exp())) * mf).sum() / nm
            for j in range(len(tails)):
                loss = loss + a.head_w * (F.binary_cross_entropy_with_logits(e[..., a.sig + j], tailt[i0, j], reduction="none") * mf).sum() / nm
            for j in range(nnb):
                vf = mf * fit_box["nbv"][i0, j].float()
                loss = loss + a.nbaux * (((e[..., a.sig + len(tails) + j] - fit_box["nbt"][i0, j]) ** 2) * vf).sum() / vf.sum().clamp(min=1.0)
        if pl_t is not None and unl_t is not None and not pre:
            pm = isd & unl_t[i0] & pl_ok[i0]
            if mk is not None:
                pm = pm & ~mk
            pf = pm.float()
            lp_ = (((p - pl_t[i0]) ** 2) * pf).sum() / pf.sum().clamp(min=1.0)
            loss = loss + a.pl_w * lp_; xls["pl"][0] += lp_.detach(); xls["pl"][1] += 1
        if (a.kd_w > 0 or a.kd_h > 0) and dr is not None and ep >= a.kd_start + a.pre_ep:
            km = dr & (m | (unl_t[i0] if (a.kd_unl and unl_t is not None) else False))
            with torch.no_grad(), torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=ADT, enabled=dev == "cuda"):
                pt, ht = fwd(ema, embed(ema, kind, idx, valid, None, None), t, valid, ctx=bool(a.ctx), return_h=True, rk=rk_of(kind, idx))
                pt = pt.float()
            kf = km.float(); nk = kf.sum().clamp(min=1.0)
            lk_ = (((p - pt) ** 2) * kf).sum() / nk
            loss = loss + a.kd_w * lk_; xls["kd"][0] += lk_.detach(); xls["kd"][1] += 1
            if a.kd_h > 0:
                dh_ = hh.shape[-1]
                lh_ = (((F.layer_norm(hh.float(), (dh_,)) - F.layer_norm(ht.float(), (dh_,))) ** 2).mean(-1) * kf).sum() / nk
                loss = loss + a.kd_h * lh_; xls.setdefault("kdh", [0.0, 0]); xls["kdh"][0] += lh_.detach(); xls["kdh"][1] += 1
        if dr is not None and a.aux_w > 0:
            tgt = xn_all[i0[..., None], blk_t].float()
            df = dr.float()
            loss = loss + a.aux_w * (((model.rec_b(hh.float()) - tgt) ** 2).mean(-1) * df).sum() / df.sum().clamp(min=1.0)
        if mk is not None:
            md = (mk & (kind == 0)).float()
            terms = [(((model.rec_d(hh.float()) - xn_all[i0].float()) ** 2).mean(-1) * md).sum() / md.sum().clamp(min=1.0)]
            if a.arr:
                ma = (mk & (kind == 1)).float(); i1 = torch.where(kind == 1, idx, 0)
                terms.append((((model.rec_a(hh.float()) - an_all[i1]) ** 2).mean(-1) * ma).sum() / ma.sum().clamp(min=1.0))
            loss = loss + a.mtm_w * sum(terms) / len(terms)
        return loss

    def fit(dsel_tr, asel_tr, lossmask, val, epochs, seed, label, unl=None, qry=None):
        """train one seed on the streams of the selected months; with `val` = (tok, bounds, rows) return the best-epoch val predictions.
        unl = (dsel, asel) label-free streams added to the training streams when --mtm > 0 (no supervised loss on them)
        qry = (tok, bounds, rows) query tokens of the --knn features (default: val). Returns model (the EMA copy under --ema), best epoch,
        {name: val predictions at the best epoch}, {knn name: values for the query rows}"""
        nonlocal step
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        model = STX(xn_all.shape[1], cards, an_all.shape[1], A["cards"], a.d, a.layers, a.heads, a.ff, a.drop,
                    tdir=a.tdir, rope=a.rope, mtm=int(a.mtm > 0 or a.pre_ep > 0),
                    n_blk=len(blk_idx) if a.adrop > 0 and a.aux_w > 0 else 0, nhead=nhead,
                    ple=a.ple, ple_d=a.ple_d, n_ple=n_ple, rel=a.rel, arch=a.arch).to(dev)
        ema = copy.deepcopy(model).eval() if a.ema > 0 else None
        for q_ in (ema.parameters() if ema is not None else ()):
            q_.requires_grad_(False)
        ev = ema if ema is not None else model
        n_upd, best_state, train_rows = 0, None, dsel_tr
        unl_t = None
        if (a.mtm > 0 or a.pre_ep > 0 or a.pl or a.kd_unl) and unl is not None:
            dsel_tr, asel_tr = dsel_tr | unl[0], asel_tr | unl[1]
            unl_t = torch.tensor(np.asarray(unl[0], bool) & ~np.asarray(lossmask, bool), device=dev)
        tok, bounds = streams(dsel_tr, asel_tr)
        lm_all = torch.tensor(lossmask, device=dev)
        fit_box["lm_all"], fit_box["unl_t"] = lm_all, unl_t
        if nnb:  # --nbaux targets from this fit's own loss rows only
            lm_np = np.asarray(lossmask, bool)
            T_, V_ = zip(*(nb_targets(lm_np, dep_t, g_, r_nb, w_, a.nb_k, a.nb_min) for w_ in nb_wins for g_ in nb_g))
            T_, V_ = np.stack(T_, 1), np.stack(V_, 1)
            fit_box["nbt"], fit_box["nbv"] = torch.tensor(T_, device=dev), torch.tensor(V_, device=dev)
            print(f"{label} nbaux: target on {V_[lm_np].mean(0).round(3).tolist()} of loss rows, sd "
                  f"{[round(float(T_[lm_np & V_[:, j], j].std()) * SCALE, 1) for j in range(nnb)]} s", flush=True)
            del T_, V_
        n_steps = sum(math.ceil((hi - lo) / L) + 1 for lo, hi in bounds) // a.bs + 1
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, fused=bool(a.fused))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=(epochs + a.pre_ep) * n_steps + 10, pct_start=a.pct)
        scaler = torch.amp.GradScaler("cuda", enabled=a.amp == "fp16" and dev == "cuda")
        best, best_ep, best_pv = np.inf, 0, None
        for ep in range(epochs + a.pre_ep):
            pre = ep < a.pre_ep
            model.train()
            st, ln = [], []
            for lo, hi in bounds:
                o = int(rng.integers(0, L)); s = np.arange(lo - o, hi, L); e = np.minimum(s + L, hi); s = np.maximum(s, lo)
                st.append(s); ln.append(e - s)
            st, ln = np.concatenate(st), np.concatenate(ln); ok = ln > 0; perm = rng.permutation(int(ok.sum()))
            st_t, ln_t = torch.tensor(st[ok][perm], device=dev), torch.tensor(ln[ok][perm], device=dev)
            tl, nb, xls = 0.0, 0, {"pl": [0.0, 0], "kd": [0.0, 0]}
            for i in range(0, min(len(st_t), n_steps * a.bs), a.bs):
                sb, lb = st_t[i:i + a.bs], ln_t[i:i + a.bs]
                if a.nosync and len(sb) < a.bs:  # static batch shape: pad with empty chunks
                    sb = torch.cat([sb, sb.new_zeros(a.bs - len(sb))]); lb = torch.cat([lb, lb.new_zeros(a.bs - len(lb))])
                kind, idx, t, valid = gather(tok, sb, lb)
                mrate = a.pre_mtm if pre else a.mtm
                mk = (valid & (torch.rand(valid.shape, device=dev) < mrate)) if mrate > 0 else None
                dr = None
                if a.adrop > 0 and not pre:
                    u = torch.rand(valid.shape, device=dev)
                    if a.adrop_chunk > 0:  # dark chunks: every tracked departure token dropped (u < 1 always)
                        u = torch.where(torch.rand(valid.shape[0], 1, device=dev) < a.adrop_chunk, torch.zeros_like(u), u)
                    dr = valid & (kind == 0) & trk_all[torch.where(kind == 0, idx, 0)] & (u < a.adrop)
                    if mk is not None:
                        dr = dr & ~mk
                with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=ADT, enabled=dev == "cuda"):
                    p, hh = fwd(model, embed(model, kind, idx, valid, mk, dr), t, valid, ctx=bool(a.ctx), return_h=True, rk=rk_of(kind, idx))
                    p = p.float()
                if a.nosync:
                    loss = static_loss(model, ema, p, hh, kind, idx, valid, t, mk, dr, pre, ep, xls)
                else:
                    m = valid & (kind == 0) & lm_all[torch.where(kind == 0, idx, 0)]
                    if pre:
                        m = m & False
                    if mk is not None:
                        m = m & ~mk
                    loss = F.mse_loss(p[m], tg[idx[m]]) if m.any() else p.sum() * 0.0
                    if m.any() and nhead:
                        e = model.head2(hh[m].float())
                        if a.sig:
                            lv = e[:, 0].clamp(-8, 4)
                            loss = loss + a.head_w * (0.5 * (lv + (tg[idx[m]] - p[m].detach()) ** 2 / lv.exp())).mean()
                        for j in range(len(tails)):
                            loss = loss + a.head_w * F.binary_cross_entropy_with_logits(e[:, a.sig + j], tailt[idx[m], j])
                        for j in range(nnb):
                            v_ = fit_box["nbv"][idx[m], j]
                            if v_.any():
                                loss = loss + a.nbaux * F.mse_loss(e[v_, a.sig + len(tails) + j], fit_box["nbt"][idx[m][v_], j])
                    xl = False  # pseudo-label / distillation terms (teacher-student)
                    if pl_t is not None and unl_t is not None and not pre:
                        i0 = torch.where(kind == 0, idx, 0)
                        pm = valid & (kind == 0) & unl_t[i0] & pl_ok[i0]
                        if mk is not None:
                            pm = pm & ~mk
                        if pm.any():
                            lp_ = F.mse_loss(p[pm], pl_t[idx[pm]]); loss = loss + a.pl_w * lp_; xl = True
                            xls["pl"][0] += lp_.item(); xls["pl"][1] += 1
                    if (a.kd_w > 0 or a.kd_h > 0) and dr is not None and ep >= a.kd_start + a.pre_ep and dr.any():
                        km = dr & (m | (unl_t[torch.where(kind == 0, idx, 0)] if (a.kd_unl and unl_t is not None) else False))
                        if km.any():
                            with torch.no_grad(), torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=ADT, enabled=dev == "cuda"):
                                pt, ht = fwd(ema, embed(ema, kind, idx, valid, None, None), t, valid, ctx=bool(a.ctx), return_h=True, rk=rk_of(kind, idx))
                                pt = pt.float()
                            lk_ = F.mse_loss(p[km], pt[km]); loss = loss + a.kd_w * lk_; xl = True
                            xls["kd"][0] += lk_.item(); xls["kd"][1] += 1
                            if a.kd_h > 0:
                                dh_ = hh.shape[-1]
                                lh_ = F.mse_loss(F.layer_norm(hh[km].float(), (dh_,)), F.layer_norm(ht[km].float(), (dh_,)))
                                loss = loss + a.kd_h * lh_; xls.setdefault("kdh", [0.0, 0]); xls["kdh"][0] += lh_.item(); xls["kdh"][1] += 1
                    if dr is not None and a.aux_w > 0 and dr.any():
                        loss = loss + a.aux_w * F.mse_loss(model.rec_b(hh[dr].float()), xn_all[idx[dr]][:, blk_idx].float())
                    if mk is not None:
                        md, ma = mk & (kind == 0), mk & (kind == 1)
                        rl = []
                        if md.any():
                            rl.append(F.mse_loss(model.rec_d(hh[md].float()), xn_all[idx[md]].float()))
                        if ma.any():
                            rl.append(F.mse_loss(model.rec_a(hh[ma].float()), an_all[idx[ma]]))
                        if rl:
                            loss = loss + a.mtm_w * sum(rl) / len(rl)
                    elif not (m.any() or xl):
                        continue
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); tl += loss.detach() if a.nosync else loss.item(); nb += 1
                if ema is not None:
                    n_upd += 1; dk = min(a.ema, (1 + n_upd) / (10 + n_upd))
                    with torch.no_grad():
                        torch._foreach_lerp_(list(ema.parameters()), [q_.detach() for q_ in model.parameters()], 1.0 - dk)
            step += 1
            if pre:
                report(run, step, total, f"{label} pretrain epoch {ep + 1}/{a.pre_ep} loss {tl / max(nb, 1):.4f}")
                continue
            ep -= a.pre_ep
            if val is None:
                report(run, step, total, f"{label} epoch {ep + 1}/{epochs} train mse {tl / max(nb, 1):.4f}")
                continue
            vtok, vb, rows = val
            pd_ = predict(ev, vtok, vb, len(y))
            pv = pd_["reg"][rows] + base[rows]
            s_all = rmse(pv, y[rows]); cl = y[rows] <= 7200; s_cl = rmse(pv[cl], y[rows][cl])
            report(run, step, total, f"{label} epoch {ep + 1}: val rmse {s_all:.2f} clean {s_cl:.2f} train mse {tl / max(nb, 1):.4f}")
            xs_ = " ".join(f"{k} {v[0] / v[1]:.4f}" for k, v in xls.items() if v[1])
            print(f"{label} ep {ep + 1} val {s_all:.2f} clean {s_cl:.2f} {xs_} slopes/h {np.round(ev.log_slope.exp().detach().cpu().numpy(), 3).tolist()}", flush=True)
            if s_all < best:
                best, best_ep = s_all, ep + 1
                best_pv = {n: (pv if n == "reg" else pd_[n][rows]) for n in out_names}
                if a.mask_val == "final":
                    best_pv["reg_masked"] = predict(ev, vtok, vb, len(y), hide=hid_t)["reg"][rows] + base[rows]
                if knns:
                    best_state = {k_: v_.clone() for k_, v_ in ev.state_dict().items()}
        kn = {}
        if knns and (qry or val):
            if best_state is not None:
                ev.load_state_dict(best_state)
            ktok, kb = streams(train_rows, asel_tr)
            kn = knn_feats(ev, qry or val, ktok, kb, np.flatnonzero(train_rows & ~np.isnan(y)))
        if a.compile:
            print(f"compile: {torch._dynamo.utils.counters['stats']['unique_graphs']} graphs so far, {sum(torch._dynamo.utils.counters['frames'].values())} frames", flush=True)
        return ev, best_ep, best_pv, kn

    res = {"id": a.id, "model": "stx", "args": vars(a), "folds": {}}
    step = 0
    allapt = np.isin(dep_apt, APTS); a_apt = np.isin(A["apt"], APTS)
    if a.refit_only:
        n_ep = int(json.loads((REP / "results" / f"{a.id}.json").read_text())["final_epochs"])
    else:
        oof, best_eps = [], []
        for fname, fd in folds:
            dtr = trm & np.isin(month, fd["train"]) & allapt
            atr = a_tr & np.isin(A["month"], fd["train"]) & a_apt
            dva = trm & np.isin(month, fd["val"]) & allapt
            ava = a_tr & np.isin(A["month"], fd["val"]) & a_apt
            vtok, vb = streams(dva, ava); rows = np.flatnonzero(dva)
            if a.mask_val == "hm":
                hid = dva & np.isin(df["MVT_ID_mvt"].to_numpy(), hide_ids()); hidden = hid[rows]
                hidx = torch.tensor(np.flatnonzero(hid), device=dev); saved_x = xn_all[hidx].clone()
                xn_all[hidx[:, None], blk_t] = unt_vec.to(xn_all.dtype)
            elif a.mask_val:
                hp = pd.Series(dep_apt).map(hrate).fillna(0.0).to_numpy()
                hid = dva & trk & (np.random.default_rng(2000 + len(res["folds"])).random(len(y)) < hp)
                hid_t = torch.tensor(hid, device=dev); hidden = hid[rows]
            pvs = []
            for s in range(a.seeds):
                _, be, pv, kn = fit(dtr, atr, dtr & ~imp, (vtok, vb, rows), a.epochs, a.seed0 + s, f"{fname} seed {s}", unl=(dva | (rank & allapt if a.unl_rank else False), ava | (a_rank & a_apt if a.unl_rank else False)))
                pvs.append({**pv, **kn}); best_eps.append(be)
            if a.mask_val == "hm":
                xn_all[hidx] = saved_x; del saved_x
            pvd = {n: np.mean([q_[n] for q_ in pvs], 0) for n in pvs[0]}; pv = pvd.pop("reg"); yv = y[rows]
            res["folds"][fname] = {"reg_rmse": rmse(pv, yv), "reg_clean_rmse": rmse(pv[yv <= 7200], yv[yv <= 7200]), "best_epochs": best_eps[-a.seeds:]}
            print(fname, res["folds"][fname], flush=True)
            oof.append(pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy()[rows], "fold": fname, "y": yv, "reg": pv, **pvd, **({"hidden": hidden} if a.mask_val else {})}))
        o = pd.concat(oof)
        res["cv_reg_rmse"] = rmse(o.reg, o.y); res["cv_reg_clean_rmse"] = rmse(o.reg[o.y <= 7200], o.y[o.y <= 7200])
        n_ep = max(2, int(round(np.mean(best_eps))))
        if a.mask_val == "hm":
            hd = o.hidden.to_numpy(bool); yy = o.y.to_numpy(); r1 = o.reg.to_numpy()
            res["mask_val"] = {"mode": "hm", "n_hidden": int(hd.sum()), "hidden_rows_masked": rmse(r1[hd], yy[hd])}
            print("mask_val hm", json.dumps(res["mask_val"]), flush=True)
        elif a.mask_val:
            ap_ = pd.Series(dep_apt[np.concatenate([np.flatnonzero(trm & np.isin(month, fd["val"]) & allapt) for _, fd in folds])]).to_numpy()
            hd = o.hidden.to_numpy(bool); yy = o.y.to_numpy(); r0, r1 = o.reg.to_numpy(), o.reg_masked.to_numpy()
            mv = {"plain": rmse(r0, yy), "masked": rmse(r1, yy), "hidden_share": float(hd.mean()),
                  "hidden_rows_plain": rmse(r0[hd], yy[hd]), "hidden_rows_masked": rmse(r1[hd], yy[hd]),
                  "hidden_rows_bias_masked": float((r1[hd] - yy[hd]).mean()), "hidden_rows_bias_plain": float((r0[hd] - yy[hd]).mean()), "by_apt": {}}
            for ap2 in sorted(set(ap_[hd])):
                k = hd & (ap_ == ap2)
                mv["by_apt"][ap2] = {"n": int(k.sum()), "plain": rmse(r0[k], yy[k]), "masked": rmse(r1[k], yy[k]),
                                     "bias_masked": float((r1[k] - yy[k]).mean()), "bias_plain": float((r0[k] - yy[k]).mean())}
            res["mask_val"] = mv
            print("mask_val", json.dumps(mv), flush=True)
        res["final_epochs"] = n_ep
        if a.smoke:
            print("smoke oof", o.describe().T.round(2), flush=True)
        else:
            o.to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    if a.final or a.refit_only:
        dtr = trm & allapt; atr = a_tr & a_apt
        te = np.flatnonzero(rank & allapt)
        ttok, tb = streams(rank & allapt, a_rank & a_apt)
        pts = []
        for s in range(a.seeds):
            m, _, _, kn = fit(dtr, atr, dtr & ~imp, None, n_ep, 100 + a.seed0 + s, f"final seed {s}", unl=(rank & allapt, a_rank & a_apt),
                              qry=(ttok, tb, te))
            pr = predict(m, ttok, tb, len(y)); pr = {n: v[te] for n, v in pr.items()}; pr["reg"] = pr["reg"] + base[te]
            pts.append({**pr, **kn})
        ot = pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy()[te], **{n: np.mean([q_[n] for q_ in pts], 0) for n in pts[0]}})
        if a.smoke:
            print("smoke test", ot.describe().T.round(2), flush=True)
            report(run, total, total, f"smoke done CV {res['cv_reg_rmse']:.2f}", finished=True)
            return
        ot.to_parquet(OUTD / "test" / f"{a.id}_raw.parquet", index=False)
    if a.refit_only:
        report(run, total, total, f"refit ({n_ep} epochs, {a.seeds} seeds) written", finished=True)
        return
    res["minutes"] = round((time.time() - t0) / 60, 2)
    save_result(res)
    report(run, total, total, f"STX reg CV {res['cv_reg_rmse']:.2f} clean {res['cv_reg_clean_rmse']:.2f} ({res['minutes']} min)", finished=True)


if __name__ == "__main__":
    main()
