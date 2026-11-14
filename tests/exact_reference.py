"""Independent exact reference solver (small n only).

Exhaustively enumerates the dual active sets and solves each reduced KKT
system with exact rational arithmetic.  The first feasible (KKT-consistent)
point is optimal because the problem is strictly convex.  Used by the test
suite to certify ``tv.denoise`` independently of the taut-string algorithm.
"""

from fractions import Fraction as F
from itertools import product

__all__ = ["solve_exact", "objective", "kkt_violation"]

# per-coordinate dual choice: -1 -> u = -p, 0 -> u = 0, +1 -> u = +p,
# FREE -> u is a free variable satisfying the stationarity equation.
_FREE = 2


def _solve_linear(mat, rhs):
    """Gauss-Jordan on an augmented k x (k+1) rational matrix."""
    k = len(mat)
    a = [row[:] + [rhs[r]] for r, row in enumerate(mat)]
    for c in range(k):
        piv = None
        for r in range(c, k):
            if a[r][c] != 0:
                piv = r
                break
        if piv is None:
            return None
        a[c], a[piv] = a[piv], a[c]
        pv = a[c][c]
        a[c] = [v / pv for v in a[c]]
        for r in range(k):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [a[r][j] - f * a[c][j] for j in range(k + 1)]
    return [a[r][k] for r in range(k)]


def solve_exact(y, weights, penalties):
    """Exact minimiser (list of Fraction) of 0.5 w(x-y)^2 + p|Dx|.

    Exponential in n; intended for n <= 8 in tests.
    """
    n = len(y)
    y = [F(v) for v in y]
    w = [F(v) for v in weights]
    pm = [F(v) for v in penalties]
    if n == 1:
        return [y[0]]
    m = n - 1

    # Dual: max_{|u_i|<=p_i} Dy.u - 0.5 * sum_i (u_i-u_{i-1})^2 / w_i,
    # with u_{-1} = u_m = 0.  Free coordinates satisfy (A u)_i = Dy_i where
    #   A_ii = 1/w_i + 1/w_{i+1},  A_{i,i+1} = -1/w_{i+1}.
    A = [[F(0)] * m for _ in range(m)]
    for i in range(m):
        A[i][i] = F(1) / w[i] + F(1) / w[i + 1]
        if i + 1 < m:
            A[i][i + 1] = -F(1) / w[i + 1]
            A[i + 1][i] = -F(1) / w[i + 1]
    Dy = [y[i + 1] - y[i] for i in range(m)]

    states = []
    for i in range(m):
        states.append([0] if pm[i] == 0 else [-1, 1, _FREE])

    for pat in product(*states):
        free = [i for i in range(m) if pat[i] == _FREE]
        u = [F(0)] * m
        for i in range(m):
            if pat[i] == -1:
                u[i] = -pm[i]
            elif pat[i] == 1:
                u[i] = pm[i]
        k = len(free)
        if k:
            mat = [[A[free[r]][free[c]] for c in range(k)] for r in range(k)]
            rhs = [
                Dy[free[r]]
                - sum(
                    A[free[r]][j] * u[j]
                    for j in range(m)
                    if pat[j] != _FREE
                )
                for r in range(k)
            ]
            sol = _solve_linear(mat, rhs)
            if sol is None:
                continue
            for j, idx in enumerate(free):
                u[idx] = sol[j]
        # Dual feasibility.
        if any(abs(u[i]) > pm[i] for i in free):
            continue
        # Stationarity: x_i = y_i + (u_i - u_{i-1})/w_i, u_{-1}=u_m=0.
        x = [F(0)] * n
        for i in range(n):
            left = u[i] if i < m else F(0)
            right = u[i - 1] if i > 0 else F(0)
            x[i] = y[i] + (left - right) / w[i]
        # Complementary slackness.
        ok = True
        for i in range(m):
            if pm[i] == 0:
                # u_i is pinned to 0 and 0*|d| is flat, so d is free.
                continue
            dx = x[i + 1] - x[i]
            if abs(u[i]) < pm[i]:
                if dx != 0:
                    ok = False
                    break
            elif dx != 0:
                if (dx > 0) != (u[i] > 0):
                    ok = False
                    break
        if ok:
            return x
    raise ArithmeticError("no feasible active set found")


def objective(y, weights, penalties, x):
    """TV functional value; accepts Fractions or floats."""
    n = len(y)
    total = sum(F(1) / 2 * weights[i] * (x[i] - y[i]) ** 2 for i in range(n))
    total += sum(penalties[i] * abs(x[i + 1] - x[i]) for i in range(n - 1))
    return total


def kkt_violation(y, weights, penalties, x, tol=1e-9):
    """Max violation of the discrete KKT system, in signal units."""
    n = len(y)
    scale = 1.0 + max((abs(v) for v in y), default=0.0)
    dx_tol = tol * scale
    u = 0.0
    worst = 0.0
    where = "stationarity"
    for i in range(n - 1):
        u += weights[i] * (x[i] - y[i])
        over = abs(u) - penalties[i]
        if over > worst:
            worst = over
            where = "dual box at %d" % i
        d = x[i + 1] - x[i]
        if d > dx_tol:
            if abs(u - penalties[i]) > worst:
                worst = abs(u - penalties[i])
                where = "up-jump complementarity at %d" % i
        elif d < -dx_tol:
            if abs(u + penalties[i]) > worst:
                worst = abs(u + penalties[i])
                where = "down-jump complementarity at %d" % i
    u += weights[n - 1] * (x[n - 1] - y[n - 1])
    if abs(u) > worst:
        worst = abs(u)
        where = "terminal stationarity"
    return worst, where
