"""QUBO builders (dimod-native) for the paper-1 optimization.

Baseline (Borowski et al., ICCS 2020):
    fqs_bqm             -- Full QUBO Solver, x_{v,j,k}, O(M N^2) variables.

Improved stack (this work):
    assignment_bqm      -- cluster-first assignment y_{c,k} (N*M vars),
                           spatio-temporal pairwise cost, capacity via
                           unbalanced penalization (zero slack qubits)
    tsp_bqm             -- per-cluster open TSP; encoding = 'onehot' or
                           'domainwall'; optional TW conflict-pair penalties
                           (zero extra qubits)

All builders return (bqm, meta). Energies are exact transforms of each
other where claimed (verified in tests_encoding.py).
"""
from __future__ import annotations

import numpy as np
import dimod


# ---------------------------------------------------------------- FQS ----
def fqs_bqm(dist: np.ndarray, M: int, A2: float = None):
    """x_{v,j,k}: vehicle v serves customer j as its k-th stop.
    Positions k = 1..K with K = N (any vehicle may serve up to N).
    Constraints (penalty A2): each customer exactly once (global one-hot);
    each (v, k) slot at most one customer.
    """
    N = dist.shape[0] - 1
    dmax = dist.max()
    if A2 is None:
        A2 = 2 * N * dmax
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    var = lambda v, j, k: f"x_{v}_{j}_{k}"
    # objective
    for v in range(M):
        for j in range(1, N + 1):
            bqm.add_linear(var(v, j, 1), dist[0, j])
            bqm.add_linear(var(v, j, N), dist[j, 0])
        for k in range(1, N):
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    if i != j:
                        bqm.add_quadratic(var(v, i, k), var(v, j, k + 1),
                                          dist[i, j])
    # each customer exactly once across vehicles and positions
    for j in range(1, N + 1):
        terms = [(var(v, j, k), 1.0) for v in range(M)
                 for k in range(1, N + 1)]
        bqm.add_linear_equality_constraint(terms, A2, -1.0)
    # each (v, k) at most one customer
    for v in range(M):
        for k in range(1, N + 1):
            g = [var(v, j, k) for j in range(1, N + 1)]
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    bqm.add_quadratic(g[a], g[b], A2)
    meta = {"N": N, "M": M, "var": var}
    return bqm, meta


def decode_fqs(sample: dict, meta) -> list:
    N, M = meta["N"], meta["M"]
    routes = []
    for v in range(M):
        r = []
        for k in range(1, N + 1):
            hits = [j for j in range(1, N + 1)
                    if sample.get(f"x_{v}_{j}_{k}", 0) == 1]
            if len(hits) == 1:
                r.append(hits[0])
            elif len(hits) > 1:
                return None                    # invalid slot
        routes.append(r)
    served = [c for r in routes for c in r]
    if sorted(served) != list(range(1, N + 1)):
        return None                            # missed/duplicated customer
    return routes


# ------------------------------------------------------- assignment ----
def assignment_bqm(D: np.ndarray, depot_cost: np.ndarray,
                   demand: np.ndarray, M: int, cap: float,
                   A: float = None, lam1: float = None, lam2: float = None,
                   gamma_depot: float = 0.5, mutual_conflicts: set = None,
                   W: float = None):
    """y_{c,k}. Pairwise spatio-temporal spread + depot pull; one-hot per
    customer; capacity via unbalanced penalization -lam1*h + lam2*h^2 with
    h_k = cap - sum d_c y_ck  (zero slack qubits).

    mutual_conflicts: undirected pairs {i, j} that are TW-infeasible as
    consecutive stops in BOTH directions; penalized with weight W when
    co-assigned (soft prior -- they may still coexist in clusters >= 3 with
    an intermediate stop, hence soft; zero extra qubits)."""
    N = D.shape[0] - 1
    Dm = D[1:, 1:]
    scale = float(Dm.mean()) if Dm.size else 1.0
    if A is None:
        A = 20 * scale
    if lam1 is None:
        lam1 = 0.05 * scale / max(cap, 1)
    if lam2 is None:
        lam2 = 2.0 * scale / max(cap, 1) ** 2
    if W is None:
        W = 5 * scale
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    var = lambda c, k: f"y_{c}_{k}"
    mc = mutual_conflicts or set()
    for k in range(M):
        for c in range(1, N + 1):
            bqm.add_linear(var(c, k), gamma_depot * depot_cost[c])
            for c2 in range(c + 1, N + 1):
                w = float(D[c, c2])
                if (c, c2) in mc or (c2, c) in mc:
                    w += W
                bqm.add_quadratic(var(c, k), var(c2, k), w)
    for c in range(1, N + 1):
        bqm.add_linear_equality_constraint(
            [(var(c, k), 1.0) for k in range(M)], A, -1.0)
    # unbalanced capacity, expanded
    for k in range(M):
        bqm.offset += -lam1 * cap + lam2 * cap ** 2
        lin = lam1 - 2 * lam2 * cap
        for c in range(1, N + 1):
            dc = float(demand[c])
            bqm.add_linear(var(c, k), lin * dc + lam2 * dc * dc)
            for c2 in range(c + 1, N + 1):
                bqm.add_quadratic(var(c, k), var(c2, k),
                                  2 * lam2 * dc * float(demand[c2]))
    return bqm, {"N": N, "M": M}


def decode_assignment(sample: dict, meta) -> list:
    N, M = meta["N"], meta["M"]
    clusters = [[] for _ in range(M)]
    for c in range(1, N + 1):
        row = [sample.get(f"y_{c}_{k}", 0) for k in range(M)]
        clusters[int(np.argmax(row))].append(c)
    return clusters


# -------------------------------------------------------------- TSP ----
def tw_conflicts(nodes, E, L, S, tau):
    """Directed pairs that can never be consecutive i->j:
    E_i + S_i + tau_ij > L_j (negated exact-rule feasibility)."""
    return {(i, j) for i in nodes for j in nodes
            if i != j and E[i] + S[i] + tau[i, j] > L[j] + 1e-9}


def _tsp_qubo_dense(nodes, tau, conflicts, Gamma, A2):
    """One-hot TSP as dense (Q, offset) over x_{p, ci}; A2 may be 0 to get
    the pure objective (used by the domain-wall transform)."""
    n = len(nodes)
    nv = n * n
    Q = np.zeros((nv, nv))
    vid = lambda p, ci: p * n + ci
    for ci, c in enumerate(nodes):
        Q[vid(0, ci), vid(0, ci)] += tau[0, c]
        Q[vid(n - 1, ci), vid(n - 1, ci)] += tau[c, 0]
    for p in range(n - 1):
        for ci, c in enumerate(nodes):
            for cj, c2 in enumerate(nodes):
                if ci != cj:
                    w = tau[c, c2]
                    if (c, c2) in conflicts:
                        w += Gamma
                    Q[vid(p, ci), vid(p + 1, cj)] += w
    off = 0.0
    if A2 > 0:
        for p in range(n):
            g = [vid(p, ci) for ci in range(n)]
            off += _add_onehot(Q, g, A2)
        for ci in range(n):
            g = [vid(p, ci) for p in range(n)]
            off += _add_onehot(Q, g, A2)
    return (Q + Q.T) / 2, off


def _add_onehot(Q, g, A):
    for a in g:
        Q[a, a] -= A
        for b in g:
            if a != b:
                Q[a, b] += A
    return A


def tsp_bqm(nodes: list, tau: np.ndarray, encoding: str = "domainwall",
            conflicts: set = None, Gamma: float = None,
            A2: float = None, kappa: float = None, A_city: float = None):
    """Open TSP over `nodes` (depot 0 implicit at both ends)."""
    n = len(nodes)
    conflicts = conflicts or set()
    sub = [0] + list(nodes)
    dmax = max(tau[i, j] for i in sub for j in sub if i != j) or 1.0
    if Gamma is None:
        Gamma = 4 * dmax
    if A2 is None:
        A2 = 2 * n * dmax
    if kappa is None:
        kappa = 4 * n * dmax
    if A_city is None:
        A_city = 2 * n * dmax
    if n == 1:
        bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
        bqm.add_linear("z_dummy", 0.0)
        return bqm, {"n": 1, "nodes": list(nodes), "encoding": encoding}

    if encoding == "onehot":
        Q, off = _tsp_qubo_dense(nodes, tau, conflicts, Gamma, A2)
        bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
        bqm.offset = off
        nv = n * n
        names = [f"x_{p}_{ci}" for p in range(n) for ci in range(n)]
        for a in range(nv):
            if Q[a, a]:
                bqm.add_linear(names[a], float(Q[a, a]))
            for b in range(a + 1, nv):
                if Q[a, b]:
                    bqm.add_quadratic(names[a], names[b], float(2 * Q[a, b]))
        return bqm, {"n": n, "nodes": list(nodes), "encoding": "onehot"}

    # domain-wall: exact affine transform of the A2 = 0 objective
    Qx, offx = _tsp_qubo_dense(nodes, tau, conflicts, Gamma, 0.0)
    nx, nz = n * n, n * (n - 1)
    B = np.zeros((nx, nz))
    b0 = np.zeros(nx)
    zid = lambda p, k: p * (n - 1) + (k - 1)          # k in 1..n-1
    for p in range(n):
        for ci in range(n):
            row = p * n + ci
            if ci == 0:
                b0[row] += 1.0
                B[row, zid(p, 1)] -= 1.0
            elif ci == n - 1:
                B[row, zid(p, n - 1)] += 1.0
            else:
                B[row, zid(p, ci)] += 1.0
                B[row, zid(p, ci + 1)] -= 1.0
    Qz = B.T @ Qx @ B
    lin = 2 * b0 @ Qx @ B
    off = offx + float(b0 @ Qx @ b0)
    Qz[np.diag_indices(nz)] += lin
    for p in range(n):                                # wall monotonicity
        for k in range(1, n - 1):
            Qz[zid(p, k + 1), zid(p, k + 1)] += kappa
            Qz[zid(p, k + 1), zid(p, k)] -= kappa / 2
            Qz[zid(p, k), zid(p, k + 1)] -= kappa / 2
    for ci in range(n):                               # each city once
        rows = [p * n + ci for p in range(n)]
        v = B[rows].sum(0)
        c0 = b0[rows].sum()
        Qz += A_city * np.outer(v, v)
        Qz[np.diag_indices(nz)] += A_city * 2 * (c0 - 1) * v
        off += A_city * (c0 - 1) ** 2
    Qz = (Qz + Qz.T) / 2
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    bqm.offset = off
    names = [f"z_{p}_{k}" for p in range(n) for k in range(1, n)]
    for a in range(nz):
        if Qz[a, a]:
            bqm.add_linear(names[a], float(Qz[a, a]))
        for b in range(a + 1, nz):
            if Qz[a, b]:
                bqm.add_quadratic(names[a], names[b], float(2 * Qz[a, b]))
    return bqm, {"n": n, "nodes": list(nodes), "encoding": "domainwall"}


def decode_tsp(sample: dict, meta) -> list | None:
    n, nodes = meta["n"], meta["nodes"]
    if n == 1:
        return list(nodes)
    order = []
    if meta["encoding"] == "onehot":
        for p in range(n):
            hits = [ci for ci in range(n) if sample.get(f"x_{p}_{ci}", 0)]
            if len(hits) != 1:
                return None
            order.append(nodes[hits[0]])
    else:
        for p in range(n):
            z = [sample.get(f"z_{p}_{k}", 0) for k in range(1, n)]
            # reject broken walls (1 after 0)
            for k in range(len(z) - 1):
                if z[k + 1] > z[k]:
                    return None
            order.append(nodes[int(sum(z))])
    if len(set(order)) != n:
        return None
    return order
