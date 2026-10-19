#!/usr/bin/env python3
"""Numerical study of Section 8 (synthetic data, fully reproducible).

All figures are written as 600-dpi PNG files (the format used in the
manuscript) and, additionally, as vector PDF files.

Experiment A (grid consistency, Theorem 5.11): a cadlag crisp step DSS with
two jumps is sampled on uniform and jittered grids of increasing size N.
The sampled variation stabilises at V(S) once the mesh separates the jumps,
while the maximal difference quotient (speed) grows like 1/mesh.

Experiment B (change localisation under observation noise): crisp step DSS
with planted change points are observed with random bit flips. Two-window
statistics are compared:
  * C_alpha : D_alpha distance between time-averaged snapshots (Def. 6.15)
              on the left and right windows, alpha in {0, 0.1, 0.25, 0.5, 0.75, 1};
  * V_win   : windowed sampled variation (sum of increments, Algorithm 1),
              alpha = 1/2.
Performance is reported as ROC-AUC for separating change locations from
locations far from any change, as mean absolute localisation error, and as
the fraction of runs in which all change points are found within one sample.

Usage:  python3 simulation.py      (from any working directory)
Outputs are written to results/simulation_results.json and to figures/
inside the repository (both folders are created automatically).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

import dss

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = HERE / "results"
RES.mkdir(parents=True, exist_ok=True)
PNG_DPI = 600


def save(fig, name):
    """Write a figure as PNG (used in the manuscript) and as vector PDF."""
    fig.savefig(OUT / f"{name}.png", dpi=PNG_DPI)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Experiment A
# ---------------------------------------------------------------------------
m, n, alpha = 3, 6, 0.5
X0 = dss.crisp([0, 1], {0: [0, 1, 2], 1: [3, 4]}, m, n)
X1 = dss.crisp([0, 1, 2], {0: [0, 1, 2], 1: [3, 4], 2: [2]}, m, n)   # activation of e3
X2 = dss.crisp([0, 1, 2], {0: [0, 1], 1: [3, 4, 5], 2: [2]}, m, n)   # value jump
taus = [0.3, 0.62]


def S_true(t):
    if t < taus[0]:
        return X0
    if t < taus[1]:
        return X1
    return X2


V_true = dss.dist(X0, X1, alpha) + dss.dist(X1, X2, alpha)
rowsA = []
for N in [5, 10, 100, 1000, 10000]:
    for kind in ["uniform", "jittered"]:
        if kind == "uniform":
            t = np.linspace(0, 1, N + 1)
        else:
            r = np.random.default_rng(N)
            inner = np.sort(r.uniform(0, 1, N - 1))
            t = np.concatenate([[0], inner, [1]])
        snaps = [S_true(x) for x in t]
        prof = dss.variation_profile(snaps, t, alpha)
        V_P = prof[-1]["cum"]
        vmax = max(p["speed"] for p in prof)
        mesh = float(np.max(np.diff(t)))
        rowsA.append(dict(N=N, grid=kind, mesh=mesh, V_P=float(V_P), max_speed=float(vmax)))
print("V(S) =", float(V_true))
for r in rowsA:
    print(r)

# ---------------------------------------------------------------------------
# Experiment B
# ---------------------------------------------------------------------------
mB, nB = 4, 20
Ngrid = 400           # samples on [0, 1]
K = 3                 # change points per run
hwin = 15             # half-window (samples)
p_flip, q_flip = 0.08, 0.02  # q_flip overridden per regime
runs = 200


def random_snapshot(r, active):
    w = np.zeros(mB)
    mu = np.zeros((mB, nB))
    for e in active:
        w[e] = 1
        mu[e] = (r.random(nB) < 0.3).astype(float)
    return w, mu


def random_change(r, X):
    w, mu = X[0].copy(), X[1].copy()
    if r.random() < 0.4:                                # structural change
        e = r.integers(mB)
        if w[e] == 1 and w.sum() == 1:                  # never deactivate the only active parameter:
            e = r.choice(np.flatnonzero(w == 0))        # activate an inactive one instead
        if w[e] == 1:
            w[e] = 0
            mu[e] = 0
        else:
            w[e] = 1
            mu[e] = (r.random(nB) < 0.3).astype(float)
        return w, mu, "structural"
    act = np.flatnonzero(w)                             # value change
    e = r.choice(act)
    idx = r.choice(nB, size=4, replace=False)
    mu[e, idx] = 1 - mu[e, idx]
    return w, mu, "value"


def observe(r, X):
    w, mu = X
    wo = w.copy()
    flip = r.random(mB) < q_flip
    wo[flip] = 1 - wo[flip]
    muo = mu.copy()
    muo[wo == 0] = 0
    noise = (r.random((mB, nB)) < p_flip) & (wo[:, None] == 1)
    muo[noise] = 1 - muo[noise]
    return wo, muo


def window_avg(snaps, a, b):
    ws = np.mean([s[0] for s in snaps[a:b]], axis=0)
    ms = np.mean([s[1] for s in snaps[a:b]], axis=0)
    return ws, ms


alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
stats_names = [f"C_{a}" for a in alphas] + ["V_win"]
regimes = {"R1 (activation noise q=0.02)": 0.02, "R2 (no activation noise, q=0)": 0.0}
resB = {}
kinds_count = {"structural": 0, "value": 0}
example_trace = None

for reg_name, qv in regimes.items():
    q_flip = qv
    rng_reg = np.random.default_rng(20261016)
    scores = {k: [] for k in stats_names}
    labels = []
    loc_err = {k: [] for k in stats_names}
    loc_max = {k: [] for k in stats_names}
    for run in range(runs):
        r = np.random.default_rng(rng_reg.integers(1 << 32))
        while True:
            cps = np.sort(r.integers(2 * hwin, Ngrid - 2 * hwin, size=K))
            if np.all(np.diff(cps) >= 3 * hwin):
                break
        X = random_snapshot(r, r.choice(mB, size=2, replace=False))
        truth = []
        cur = X
        j = 0
        for i in range(Ngrid):
            if j < K and i == cps[j]:
                w, mu, kind = random_change(r, cur)
                cur = (w, mu)
                if qv > 0:
                    kinds_count[kind] += 1
                j += 1
            truth.append(cur)
        obs = [observe(r, X) for X in truth]
        idx = np.arange(hwin, Ngrid - hwin + 1)
        st = {k: np.zeros(len(idx)) for k in stats_names}
        inc = np.array([0.0] + [dss.dist(obs[k], obs[k - 1], 0.5) for k in range(1, Ngrid)])
        for jj, i in enumerate(idx):
            L = window_avg(obs, i - hwin, i)
            R = window_avg(obs, i, i + hwin)
            P, Q = dss.dist_parts(L, R)
            for a in alphas:
                st[f"C_{a}"][jj] = a * P + (1 - a) * Q
            st["V_win"][jj] = inc[i - hwin + 1:i + hwin].sum()
        dmin = np.min(np.abs(idx[:, None] - cps[None, :]), axis=1)
        pos = dmin <= 1
        neg = dmin >= hwin
        keep = pos | neg
        labels.append(pos[keep].astype(int))
        for k in stats_names:
            scores[k].append(st[k][keep])
            s_ = st[k].copy()
            found = []
            for _ in range(K):
                p = int(np.argmax(s_))
                found.append(idx[p])
                s_[max(0, p - hwin):p + hwin + 1] = -np.inf
            found = np.sort(found)
            loc_err[k].append(np.mean(np.abs(found - cps)))
            loc_max[k].append(np.max(np.abs(found - cps)))
        if run == 0 and qv > 0:
            example_trace = dict(idx=idx.tolist(), cps=cps.tolist(),
                                 **{k: st[k].tolist() for k in stats_names})
    labels_all = np.concatenate(labels)
    res = {}
    for k in stats_names:
        auc_runs = [roc_auc_score(lab, sc) for lab, sc in zip(labels, scores[k])]
        res[k] = dict(auc_pooled=float(roc_auc_score(labels_all, np.concatenate(scores[k]))),
                      auc_mean=float(np.mean(auc_runs)), auc_sd=float(np.std(auc_runs)),
                      loc_mae=float(np.mean(loc_err[k])), loc_mae_sd=float(np.std(loc_err[k])),
                      exact_rate=float(np.mean(np.array(loc_max[k]) <= 1.0)))
    resB[reg_name] = res
    print(reg_name)
    for k in stats_names:
        print(f"  {k:8s} AUC {res[k]['auc_mean']:.3f} +- {res[k]['auc_sd']:.3f}  "
              f"MAE {res[k]['loc_mae']:.2f} +- {res[k]['loc_mae_sd']:.2f}  exact {res[k]['exact_rate']:.3f}")
print("change types (regime R1):", kinds_count)

with open(RES / "simulation_results.json", "w") as f:
    json.dump(dict(V_true=float(V_true), expA=rowsA, expB=resB, kinds=kinds_count,
                   settings=dict(m=mB, n=nB, Ngrid=Ngrid, K=K, hwin=hwin, p_flip=p_flip,
                                 q_values=list(regimes.values()), runs=runs, alphas=alphas)), f, indent=1)

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
uni = [r for r in rowsA if r["grid"] == "uniform"]
jit = [r for r in rowsA if r["grid"] == "jittered"]
ax[0].semilogx([r["N"] for r in uni], [r["V_P"] for r in uni], "o-", color="#1f4e79", label="uniform grid")
ax[0].semilogx([r["N"] for r in jit], [r["V_P"] for r in jit], "s--", color="#c55a11", label="jittered grid")
ax[0].axhline(V_true, color="0.4", lw=0.8, ls=":", label=r"$V(\mathcal{S})$")
ax[0].set_xlabel("number of sampling intervals $N$")
ax[0].set_ylabel("sampled variation $V_P$")
ax[0].set_ylim(0, 0.35)
ax[0].legend(frameon=False, fontsize=7)
ax[1].loglog([r["N"] for r in uni], [r["max_speed"] for r in uni], "o-", color="#1f4e79", label="uniform grid")
ax[1].loglog([r["N"] for r in jit], [r["max_speed"] for r in jit], "s--", color="#c55a11", label="jittered grid")
ax[1].set_xlabel("number of sampling intervals $N$")
ax[1].set_ylabel(r"max difference quotient")
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "fig_grid_consistency")

fig, ax = plt.subplots(figsize=(7.0, 2.6))
tr = example_trace
x = np.array(tr["idx"]) / Ngrid
ax.plot(x, tr["C_0.0"], color="#1f4e79", lw=1.1, label=r"$C_{0}$")
ax.plot(x, tr["C_0.5"], color="#2e8b57", lw=0.9, label=r"$C_{1/2}$")
ax.plot(x, np.array(tr["V_win"]) / max(tr["V_win"]) * max(tr["C_0.0"]), color="#c55a11", lw=0.8,
        alpha=0.8, label=r"$V_{\mathrm{win}}$ (rescaled)")
for c in tr["cps"]:
    ax.axvline(c / Ngrid, color="0.3", ls=":", lw=0.9)
ax.set_xlabel("time $t$")
ax.set_ylabel("statistic")
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
fig.tight_layout()
save(fig, "fig_localisation_trace")

# graded example trajectory
t = np.linspace(0, 1, 201)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax[0].plot(t, np.ones_like(t), color="#1f4e79", label=r"$w_t(e_1)$")
ax[0].plot(t, t, color="#c55a11", label=r"$w_t(e_2)$")
ax[0].plot(t, 1 - t / 2, "--", color="#1f4e79", lw=0.9, label=r"$\mu_t(e_1,u_1)$")
ax[0].plot(t, t / 2, ":", color="#1f4e79", lw=0.9, label=r"$\mu_t(e_1,u_2)$")
ax[0].plot(t, t ** 2, "--", color="#c55a11", lw=0.9, label=r"$\mu_t(e_2,u_1)$")
ax[0].plot(t, t / 2, "-.", color="#c55a11", lw=0.9, label=r"$\mu_t(e_2,u_2)$")
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("degree")
ax[0].legend(frameon=False, fontsize=6, ncol=2)
ax[1].plot(t, 7 / 16 + t / 4, color="#1f4e79", label=r"$|\mathcal{S}'|(t)$")
ax[1].plot(t, 2 * t * (7 / 16 + t ** 2 / 4), "--", color="#c55a11",
           label=r"$|(\mathcal{S}\circ\varphi)'|(t)$, $\varphi(t)=t^2$")
ax[1].set_xlabel("$t$")
ax[1].set_ylabel("metric speed")
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "fig_graded_example")
print("figures written")
