#!/usr/bin/env python3
"""Reproduces every number quoted in the worked examples of the manuscript.

Usage:  python3 verify_examples.py      (from any working directory)
The script prints one line per check and exits with status 0 if all checks
pass and with status 1 otherwise.
"""
import itertools
import sys
from fractions import Fraction as Fr

import numpy as np

import dss

np.set_printoptions(precision=6, suppress=True)
ok = True


def check(label, val, target, tol=1e-12):
    global ok
    good = abs(float(val) - float(target)) < tol
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] {label}: {float(val):.10g}  (expected {target})")


# ---------------------------------------------------------------------------
print("\n== Proposition 4.4: union normalisation violates the triangle inequality ==")
# m = |E| parameters, n = |U| = 1; snapshots S, G, H of Proposition 4.4
thr_ok = True
for m in [2, 3, 5, 10, 50]:
    n = 1
    S = (set(), {})
    G = ({1}, {1: set()})
    H = ({0}, {0: {0}})
    thr = m / (m + 4)
    for a in [0.0, thr * 0.999, thr * 1.001, 1.0]:
        dSH = dss.old_dist(*S, *H, a, m, n)
        dSG = dss.old_dist(*S, *G, a, m, n)
        dGH = dss.old_dist(*G, *H, a, m, n)
        viol = dSH > dSG + dGH + 1e-15
        if viol != (a < thr):
            thr_ok = ok = False
            print("FAIL", m, a)
    # closed forms at alpha = 0.3
    a = 0.3
    check(f"m={m}, alpha=0.3: d(S,H)", dss.old_dist(*S, *H, a, m, n), a / m + 1 - a)
    check(f"m={m}, alpha=0.3: d(S,G)", dss.old_dist(*S, *G, a, m, n), a / m)
    check(f"m={m}, alpha=0.3: d(G,H)", dss.old_dist(*G, *H, a, m, n), 2 * a / m + (1 - a) / 2)
print("violation iff alpha < m/(m+4) for m in {2, 3, 5, 10, 50}:", "confirmed" if thr_ok else "FAILED")
# alpha = 1/2 already fails for m = |E| = 5
m, n = 5, 1
a = 0.5
lhs = dss.old_dist(set(), {}, {0}, {0: {0}}, a, m, n)
rhs = dss.old_dist(set(), {}, {1}, {1: set()}, a, m, n) + dss.old_dist({1}, {1: set()}, {0}, {0: {0}}, a, m, n)
check("alpha=1/2, m=5: d(S,H)", lhs, 0.6)
check("alpha=1/2, m=5: d(S,G) + d(G,H)", rhs, 0.55)

# ---------------------------------------------------------------------------
print("\n== Theorem 4.2: exhaustive metric check for the fixed-coordinate distance ==")
# all crisp snapshots over |E|=2, |U|=2
m, n = 2, 2
snaps = []
for act in itertools.product([0, 1], repeat=m):
    choices = []
    for e in range(m):
        if act[e]:
            choices.append(list(itertools.product([0, 1], repeat=n)))
        else:
            choices.append([(0,) * n])
    for rows in itertools.product(*choices):
        snaps.append((np.array(act, float), np.array(rows, float)))
print("number of crisp snapshots:", len(snaps), "(expected (1+2^n)^m =", (1 + 2 ** n) ** m, ")")
for a in [0.05, 1 / 3, 0.5, 0.9]:
    worst = 0.0
    for X, Y, Z in itertools.product(snaps, repeat=3):
        worst = max(worst, dss.dist(X, Z, a) - dss.dist(X, Y, a) - dss.dist(Y, Z, a))
    mins = min(dss.dist(X, Y, a) for X, Y in itertools.combinations(snaps, 2))
    gap = min(a / m, (1 - a) / (m * n))
    check(f"alpha={a:.3f} max triangle excess <= 0", max(worst, 0), 0)
    check(f"alpha={a:.3f} minimum positive gap", mins, gap)

# Lipschitz constants of the operations (Proposition 4.7)
for a in [0.2, 0.5, 0.8]:
    r_u = r_i = r_c = 0.0
    for X, Y in itertools.combinations(snaps, 2):
        d = dss.dist(X, Y, a)
        r_c = max(r_c, dss.dist(dss.complement(X), dss.complement(Y), a) / d)
    for X, X2, Y, Y2 in itertools.product(snaps[::2], repeat=4):
        den = dss.dist(X, X2, a) + dss.dist(Y, Y2, a)
        if den > 0:
            r_u = max(r_u, dss.dist(dss.union(X, Y), dss.union(X2, Y2), a) / den)
            r_i = max(r_i, dss.dist(dss.intersection(X, Y), dss.intersection(X2, Y2), a) / den)
    print(f"alpha={a}: max ratio union {r_u:.4f} (<=1), intersection {r_i:.4f} (<=1), "
          f"complement {r_c:.4f} (<= 1/alpha = {1 / a:.4f})")
    ok &= r_u <= 1 + 1e-12 and r_i <= 1 + 1e-12 and r_c <= 1 / a + 1e-12

# ---------------------------------------------------------------------------
print("\n== Example 5.9: discontinuity taxonomy (|U|=3, |E|=4, alpha=1/2) ==")
m, n, a = 4, 3, 0.5
Lm = dss.crisp([0, 1, 2], {0: [0], 1: [0, 1], 2: [1, 2]}, m, n)
S1 = dss.crisp([0, 1], {0: [0], 1: [0, 1]}, m, n)
Lp = dss.crisp([0, 1, 3], {0: [0], 1: [0, 1], 3: [2]}, m, n)
jm = dss.dist(Lm, S1, a)
jp = dss.dist(S1, Lp, a)
check("j^-(1)", jm, 1 / 8 + 1 / 12)
check("j^+(1)", jp, 1 / 8 + 1 / 24)
check("V_0^2", jm + jp, Fr(1, 8) + Fr(1, 12) + Fr(1, 8) + Fr(1, 24))
print("   exact: j^- =", Fr(1, 8) + Fr(1, 12), " j^+ =", Fr(1, 8) + Fr(1, 24), " V =",
      Fr(1, 8) + Fr(1, 12) + Fr(1, 8) + Fr(1, 24))

# ---------------------------------------------------------------------------
print("\n== Example 8.1 and Table 1: synthetic ward-type table, alpha = 1/2 ==")
m, n, a = 3, 6, 0.5
u = lambda *k: [i - 1 for i in k]
ward = [
    dss.crisp([0, 1], {0: u(1, 2, 3), 1: u(4, 5)}, m, n),
    dss.crisp([0, 1], {0: u(1, 2), 1: u(4, 5, 6)}, m, n),
    dss.crisp([0, 1, 2], {0: u(1, 2), 1: u(4, 5, 6), 2: u(3)}, m, n),
    dss.crisp([0, 1, 2], {0: u(1), 1: u(5, 6), 2: u(3, 4)}, m, n),
    dss.crisp([0, 1, 2], {0: u(1), 1: u(5, 6), 2: u(3, 4)}, m, n),
]
prof = dss.variation_profile(ward, [0, 1, 2, 3, 4], a)
targets = [Fr(1, 18), Fr(7, 36), Fr(1, 12), Fr(0)]
# columns act, deact, exp, con, P, Q of Table 1
table1 = [(0, 0, 1, 1, Fr(0), Fr(1, 9)), (1, 0, 1, 0, Fr(1, 3), Fr(1, 18)),
          (0, 0, 1, 2, Fr(0), Fr(1, 6)), (0, 0, 0, 0, Fr(0), Fr(0))]
for p, t, row in zip(prof, targets, table1):
    print("   step", p["step"], "act", p["act"], "deact", p["deact"], "exp", p["expansion"],
          "con", p["contraction"], "P", round(p["P"], 6), "Q", round(p["Q"], 6))
    got = (p["act"], p["deact"], p["expansion"], p["contraction"], p["P"], p["Q"])
    good = all(abs(float(g) - float(r)) < 1e-12 for g, r in zip(got, row))
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}]    Table 1 row {p['step']}: act, deact, exp, con, P, Q")
    check(f"   d on {p['step']}", p["d"], t)
check("total sampled variation", prof[-1]["cum"], Fr(1, 3))
avg = dss.time_average(ward, [0, 1, 2, 3, 4])
print("   time-average activation degrees:", avg[0])
print("   time-average memberships:\n", avg[1])
check("avg w(e3)", avg[0][2], 0.5)
check("avg mu(e1,u1)", avg[1][0, 0], 1.0)
check("avg mu(e1,u2)", avg[1][0, 1], 0.75)
check("avg mu(e1,u3)", avg[1][0, 2], 0.25)
check("avg mu(e2,u4)", avg[1][1, 3], 0.75)
check("avg mu(e2,u6)", avg[1][1, 5], 0.75)
check("avg mu(e3,u3)", avg[1][2, 2], 0.5)
check("avg mu(e3,u4)", avg[1][2, 3], 0.25)
print("   admissible:", dss.is_admissible(avg))

# ---------------------------------------------------------------------------
print("\n== Examples 6.16 and 6.17: graded trajectory and chain rule, |U|=|E|=2, alpha=1/2 ==")
a = 0.5


def S(t):
    w = np.array([1.0, t])
    mu = np.array([[1 - t / 2, t / 2], [t ** 2, t / 2]])
    return w, mu


ts = np.linspace(0, 1, 2001)
print("   admissible on [0,1]:", all(dss.is_admissible(S(t)) for t in ts))
# speed formula 7/16 + t/4 checked by difference quotients
for t0 in [0.1, 0.25, 0.5, 0.9]:
    h = 1e-6
    q = dss.dist(S(t0 + h), S(t0), a) / h
    check(f"   metric speed at t={t0}", q, 7 / 16 + t0 / 4, tol=1e-5)
L = sum(dss.dist(S(ts[i]), S(ts[i - 1]), a) for i in range(1, len(ts)))
check("   length on [0,1]", L, 9 / 16, tol=1e-9)
check("   D(S_0,S_1)", dss.dist(S(0), S(1), a), 9 / 16)
# chain rule phi(s) = s^2
for s0 in [0.5, 1.0 - 1e-6]:
    h = 1e-7
    q = dss.dist(S((s0 + h) ** 2), S(s0 ** 2), a) / h
    check(f"   speed of S o phi at s={s0:.3f}", q, 2 * s0 * (7 / 16 + s0 ** 2 / 4), tol=1e-5)
# layer-cake identity of Theorem 6.3(iii): D(X,Y) = int_0^1 d(X_theta, Y_theta) dtheta
X, Y = S(0.3), S(0.8)
th = (np.arange(200000) + 0.5) / 200000
lc = np.mean([dss.dist(dss.level(X, t), dss.level(Y, t), a) for t in th[::50]])
check("   layer-cake (coarse quadrature)", lc, dss.dist(X, Y, a), tol=1e-4)
# exact layer-cake: piecewise-constant in theta, breakpoints at coordinate values
vals = np.unique(np.concatenate([[0, 1], X[0], Y[0], X[1].ravel(), Y[1].ravel()]))
exact = sum((vals[i + 1] - vals[i]) * dss.dist(dss.level(X, (vals[i] + vals[i + 1]) / 2),
                                                dss.level(Y, (vals[i] + vals[i + 1]) / 2), a)
            for i in range(len(vals) - 1))
check("   layer-cake (exact)", exact, dss.dist(X, Y, a))

# ---------------------------------------------------------------------------
print("\n== Example 6.18: graded sandwich ==")
sw = True
for t in np.linspace(-1, 1, 100001):
    if t == 0:
        continue
    s = 0.5 - t * t / 2
    g = 0.5 + (t * t / 2) * np.sin(1 / t)
    hh = 0.5 + t * t / 2
    sw &= (s <= g <= hh) and 0 <= s and hh <= 1
ok &= sw
print("   ordering and admissibility hold on a 1e5-point grid of [-1,1]:", sw)

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
