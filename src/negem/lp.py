"""Minimal sparse LP/MIP builder on top of SciPy's HiGHS interface.

Two solves per horizon, mirroring commercial PCMs (PLEXOS/PROMOD):
  1. MIP  -> unit commitment (integer on-line counts per clustered unit group)
  2. LP   -> commitment fixed, re-solve to read shadow prices (LMPs, gas price, reserve price)
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint, Bounds, linprog, milp


class LP:
    def __init__(self):
        self.lb, self.ub, self.c, self.integ = [], [], [], []
        self.rows, self.cols, self.vals = [], [], []
        self.rlo, self.rhi = [], []
        self.s_r, self.s_c, self.s_v = [], [], []
        self.n = 0
        self.m = 0

    # ---- variables -----------------------------------------------------------------
    def var(self, shape, lb=0.0, ub=np.inf, cost=0.0, integer=False) -> np.ndarray:
        size = int(np.prod(shape))
        idx = np.arange(self.n, self.n + size).reshape(shape)
        self.n += size
        self.lb.append(np.broadcast_to(np.asarray(lb, float), shape).ravel())
        self.ub.append(np.broadcast_to(np.asarray(ub, float), shape).ravel())
        self.c.append(np.broadcast_to(np.asarray(cost, float), shape).ravel())
        self.integ.append(np.full(size, 1 if integer else 0))
        return idx

    # ---- constraints: sum(coef*var) in [lo, hi] -----------------------------------
    def con(self, terms, lo=-np.inf, hi=np.inf) -> int:
        """terms: list of (var_index_array_or_int, coef_array_or_scalar). Returns row id."""
        r = self.m
        for v, a in terms:
            if isinstance(v, (int, np.integer)):
                self.s_r.append(r); self.s_c.append(int(v)); self.s_v.append(float(a))
                continue
            v = np.asarray(v).ravel()
            a = np.asarray(a, float)
            a = np.full(v.size, float(a)) if a.ndim == 0 else a.ravel()
            self.rows.append(np.full(v.size, r))
            self.cols.append(v)
            self.vals.append(a)
        self.rlo.append(lo)
        self.rhi.append(hi)
        self.m += 1
        return r

    def build(self):
        self._lb = np.concatenate(self.lb)
        self._ub = np.concatenate(self.ub)
        self._c = np.concatenate(self.c)
        self._int = np.concatenate(self.integ)
        A = sparse.csr_matrix((np.concatenate(self.vals + [np.array(self.s_v)]),
                               (np.concatenate(self.rows + [np.array(self.s_r, dtype=int)]),
                                np.concatenate(self.cols + [np.array(self.s_c, dtype=int)]))),
                              shape=(self.m, self.n))
        self._A = A
        self._rlo = np.array(self.rlo, float)
        self._rhi = np.array(self.rhi, float)

    def solve_mip(self, gap=0.008, time_limit=60, rebuild=True):
        if rebuild:
            self.build()
        res = milp(self._c, integrality=self._int,
                   bounds=Bounds(self._lb, self._ub),
                   constraints=LinearConstraint(self._A, self._rlo, self._rhi),
                   options={"mip_rel_gap": gap, "time_limit": time_limit, "disp": False})
        if res.x is None:
            raise RuntimeError(f"MIP failed: {res.message}")
        return res.x, res.fun

    def solve_lp_fixed(self, x_mip, relax=None):
        """Fix integer variables at MIP values, solve LP, return primal x and row duals.

        `relax`: integer variables left continuous in the pricing run. Used for fast-start
        units (ISO-NE fast-start pricing), so a quick-start peaker that could be dispatched
        sets the price instead of a scarcity artifact of the fixed commitment.
        """
        lb, ub = self._lb.copy(), self._ub.copy()
        ints = self._int == 1
        if relax is not None and len(relax):
            ints[np.asarray(relax)] = False
        lb[ints] = ub[ints] = np.round(x_mip[ints])
        eq = np.isclose(self._rlo, self._rhi)
        up = ~eq & np.isfinite(self._rhi)
        lo = ~eq & np.isfinite(self._rlo)
        A_ub = sparse.vstack([self._A[up], -self._A[lo]]).tocsr()
        b_ub = np.concatenate([self._rhi[up], -self._rlo[lo]])
        res = linprog(self._c, A_ub=A_ub, b_ub=b_ub, A_eq=self._A[eq], b_eq=self._rhi[eq],
                      bounds=np.column_stack([lb, ub]), method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        dual = np.zeros(self.m)
        dual[np.where(eq)[0]] = res.eqlin.marginals
        n_up = up.sum()
        dual[np.where(up)[0]] += res.ineqlin.marginals[:n_up]
        dual[np.where(lo)[0]] -= res.ineqlin.marginals[n_up:]
        return res.x, dual
