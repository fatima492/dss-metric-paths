#!/usr/bin/env python3
"""
dss.py -- reference implementation for
"Dynamic soft sets as metric paths from crisp rigidity and bounded
 variation to graded differential calculus".

A snapshot over (U, E) with |E| = m, |U| = n is encoded as a pair
    (w, mu),  w in [0,1]^m,  mu in [0,1]^{m x n},  mu[e, u] <= w[e].
Crisp snapshots are the {0,1}-valued encodings: w = indicator of the
active set A, mu[e, :] = indicator of F(e) for e in A and 0 otherwise.

All distances use the fixed-coordinate normalisation of Definition 4.1 and Eq. (12):
    D_alpha(X, Y) = alpha/m * sum_e |w - w'| + (1-alpha)/(m n) * sum_{e,u} |mu - mu'|.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# encoding helpers
# ----------------------------------------------------------------------------
def crisp(active, F, m, n):
    """Encode a crisp snapshot.

    active : iterable of parameter indices (0-based) that are active
    F      : dict {e: iterable of element indices} for e in active
    """
    w = np.zeros(m)
    mu = np.zeros((m, n))
    for e in active:
        w[e] = 1.0
        for u in F.get(e, ()):
            mu[e, u] = 1.0
    return w, mu


def is_admissible(X, tol=1e-12):
    w, mu = X
    return (np.all(w >= -tol) and np.all(w <= 1 + tol) and np.all(mu >= -tol)
            and np.all(mu <= w[:, None] + tol))


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def dist(X, Y, alpha):
    """Fixed-coordinate distance d_alpha (Definition 4.1) and D_alpha (Section 6.1)."""
    (w, mu), (v, nu) = X, Y
    m, n = mu.shape
    return alpha / m * np.abs(w - v).sum() + (1 - alpha) / (m * n) * np.abs(mu - nu).sum()


def dist_parts(X, Y):
    """Return (parameter part, approximation part) before alpha-weighting:
    P = |A triangle B| / m  and  Q = sum_e |F~(e) triangle G~(e)| / (m n)."""
    (w, mu), (v, nu) = X, Y
    m, n = mu.shape
    return np.abs(w - v).sum() / m, np.abs(mu - nu).sum() / (m * n)


def old_dist(A, F, B, G, alpha, m, n):
    """The pair-dependent union normalisation |A u B| of Eq. (6) (Proposition 4.4).
    A, B: sets of parameters; F, G: dict e -> set of elements."""
    sym = len(A ^ B) / m
    union = A | B
    if not union:
        return alpha * sym
    tot = 0.0
    for e in union:
        Fe = set(F.get(e, ())) if e in A else set()
        Ge = set(G.get(e, ())) if e in B else set()
        tot += len(Fe ^ Ge) / n
    return alpha * sym + (1 - alpha) * tot / len(union)


# ----------------------------------------------------------------------------
# operations (Definition 4.6 and Eq. (14))
# ----------------------------------------------------------------------------
def union(X, Y):
    return np.maximum(X[0], Y[0]), np.maximum(X[1], Y[1])


def intersection(X, Y):
    return np.minimum(X[0], Y[0]), np.minimum(X[1], Y[1])


def complement(X):
    w, mu = X
    return w.copy(), w[:, None] - mu


# ----------------------------------------------------------------------------
# sampled-variation profile (Algorithm 1)
# ----------------------------------------------------------------------------
def variation_profile(snaps, times, alpha):
    """For consecutive snapshots return a list of dicts with the increments
    of Algorithm 1: activations, deactivations, expansion, contraction,
    parameter part, approximation part, increment d_i, speed d_i/dt and
    cumulative variation."""
    out = []
    cum = 0.0
    for i in range(1, len(snaps)):
        (w0, mu0), (w1, mu1) = snaps[i - 1], snaps[i]
        m, n = mu0.shape
        act = np.clip(w1 - w0, 0, None).sum()
        deact = np.clip(w0 - w1, 0, None).sum()
        exp_ = np.clip(mu1 - mu0, 0, None).sum()
        con = np.clip(mu0 - mu1, 0, None).sum()
        P = (act + deact) / m
        Q = (exp_ + con) / (m * n)
        d = alpha * P + (1 - alpha) * Q
        dt = times[i] - times[i - 1]
        cum += d
        out.append(dict(step=(times[i - 1], times[i]), act=act, deact=deact,
                        expansion=exp_, contraction=con, P=P, Q=Q, d=d,
                        speed=d / dt, cum=cum))
    return out


def time_average(snaps, times):
    """Time-average (Definition 6.13) of a right-continuous step path that
    holds snaps[i] on [times[i], times[i+1]); the last snapshot is the value
    at the terminal time and carries no duration."""
    T = times[-1] - times[0]
    w = sum((times[i + 1] - times[i]) * snaps[i][0] for i in range(len(snaps) - 1)) / T
    mu = sum((times[i + 1] - times[i]) * snaps[i][1] for i in range(len(snaps) - 1)) / T
    return w, mu


def level(X, theta):
    """theta-level crisp snapshot of a graded snapshot (Theorem 6.3)."""
    w, mu = X
    return (w >= theta).astype(float), (mu >= theta).astype(float)


# ----------------------------------------------------------------------------
# self-test (run "python3 dss.py")
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from fractions import Fraction

    m, n, alpha = 3, 6, 0.5
    X0 = crisp([0, 1], {0: [0, 1, 2], 1: [3, 4]}, m, n)
    X1 = crisp([0, 1], {0: [0, 1], 1: [3, 4, 5]}, m, n)
    X2 = crisp([0, 1, 2], {0: [0, 1], 1: [3, 4, 5], 2: [2]}, m, n)
    d01, d12 = dist(X0, X1, alpha), dist(X1, X2, alpha)
    prof = variation_profile([X0, X1, X2], [0.0, 1.0, 2.0], alpha)
    avg = time_average([X0, X1, X2], [0.0, 1.0, 2.0])
    checks = [
        ("d(X0, X1) = 1/18", abs(d01 - 1 / 18) < 1e-12),
        ("d(X1, X2) = 7/36", abs(d12 - 7 / 36) < 1e-12),
        ("sampled variation = 1/4", abs(prof[-1]["cum"] - 0.25) < 1e-12),
        ("parts recombine to d", abs(alpha * dist_parts(X1, X2)[0]
                                     + (1 - alpha) * dist_parts(X1, X2)[1] - d12) < 1e-12),
        ("complement is an isometry on a fixed active set",
         abs(dist(complement(X0), complement(X1), alpha) - d01) < 1e-12),
        ("time average is admissible", bool(is_admissible(avg))),
        ("level snapshot at theta=1 of a crisp snapshot is itself",
         all(np.array_equal(a, b) for a, b in zip(level(X2, 1.0), X2))),
    ]
    for label, good in checks:
        print(f"[{'OK' if good else 'FAIL'}] {label}")
    print("d(X0,X1) =", Fraction(d01).limit_denominator(1000),
          " d(X1,X2) =", Fraction(d12).limit_denominator(1000))
    raise SystemExit(0 if all(g for _, g in checks) else 1)
