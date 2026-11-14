"""Weighted one-dimensional total-variation denoising.

For a sampling of length ``n``, return the unique minimiser of

    F(x) = 0.5 * sum_i weights[i] * (x[i] - y[i])**2
         +       sum_i penalties[i] * |x[i+1] - x[i]| .

The functional is strictly convex (weights are positive), so the minimiser
exists and is unique.  Inputs must satisfy

    0 <= n <= 200000
    |y[i]| <= 10000
    0.1 <= weights[i] <= 10
    0 <= penalties[i] <= 10000

and the three arguments must be lists of length n, n and max(n-1, 0) holding
finite ints/floats (booleans are rejected).  Wrong input raises ``ValueError``
and never mutates the caller's lists.

Algorithm
---------
Weighted taut string (Davies-Kovac / Duembgen, in the linear-time form of
Condat).  Write

    tau[0] = 0,  tau[i] = sum_{j<i} weights[j]      (strictly increasing)
    S[0]   = 0,  S[i]   = sum_{j<i} weights[j] * y[j]

and let Gamma be the shortest path (Euclidean length) from (tau[0], S[0]) to
(tau[n], S[n]) that stays inside the tube

    S[i] - penalties[i-1] <= Gamma(tau[i]) <= S[i] + penalties[i-1]

at every interior node 0 < i < n (the endpoints are pinned to S[0], S[n]).
It is a classical duality result for this problem that the slopes of that
geodesic, divided by the interval widths, are the minimiser:

    x[i] = (Gamma(tau[i+1]) - Gamma(tau[i])) / weights[i] .

Intuitively the string is pulled taut: wherever it is free it is straight
(x constant, no jump), and it leaves the tube boundary only where the data
pull harder than the local penalty allows, which is exactly where a step of
the denoised signal survives.

The implementation keeps two monotone convex hulls (lower and upper) on
stacks; when they cross, the contact node becomes a knot of the taut string.
Every index is pushed once and popped at most once, so the whole computation
is linear.

Complexity: O(n) time, O(n) auxiliary space in the worst case (no prefix
rescan, no matrix).  The implementation stores its working arrays in
``array`` buffers, keeping the auxiliary footprint at about 68 bytes per
sample plus the output list.
"""

from array import array
from math import isfinite

__all__ = ["denoise"]

_MAX_N = 200_000
_Y_ABS = 10_000.0
_W_LO = 0.1
_W_HI = 10.0
_P_LO = 0.0
_P_HI = 10_000.0

_INF = float("inf")


def _fail(msg):
    raise ValueError(msg)


def _check_values(name, seq, lo, hi):
    for i, v in enumerate(seq):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _fail("%s[%d] must be an int or float, got %r" % (name, i, v))
        if isinstance(v, float) and not isfinite(v):
            _fail("%s[%d] must be finite, got %r" % (name, i, v))
        if v < lo or v > hi:
            _fail("%s[%d] is outside [%r, %r]: %r" % (name, i, lo, hi, v))


def _validate(y, weights, penalties):
    if not isinstance(y, list):
        _fail("y must be a list, got %s" % type(y).__name__)
    if not isinstance(weights, list):
        _fail("weights must be a list, got %s" % type(weights).__name__)
    if not isinstance(penalties, list):
        _fail("penalties must be a list, got %s" % type(penalties).__name__)

    n = len(y)
    if n > _MAX_N:
        _fail("n=%d exceeds the supported maximum %d" % (n, _MAX_N))
    if len(weights) != n:
        _fail("weights has length %d, expected %d" % (len(weights), n))
    m = n - 1 if n else 0
    if len(penalties) != m:
        _fail("penalties has length %d, expected %d" % (len(penalties), m))

    _check_values("y", y, -_Y_ABS, _Y_ABS)
    _check_values("weights", weights, _W_LO, _W_HI)
    _check_values("penalties", penalties, _P_LO, _P_HI)


def denoise(y, weights, penalties):
    """Return the minimiser of the weighted 1-D TV functional.

    Parameters
    ----------
    y : list of int/float, length n, 0 <= n <= 200000, |y[i]| <= 10000
    weights : list of int/float, length n, 0.1 <= weights[i] <= 10
    penalties : list of int/float, length max(n-1, 0), 0 <= penalties[i] <= 10000

    Returns
    -------
    list of float, length n (``[]`` when ``n == 0``)

    Raises
    ------
    ValueError
        On a non-list argument, a length mismatch, a boolean, a non-finite
        number, or a value outside the documented range.
    """
    _validate(y, weights, penalties)
    n = len(y)
    if n == 0:
        return []
    if n == 1:
        return [float(y[0])]
    return _taut_string(y, weights, penalties)


def _taut_string(y, w, p):
    """Linear weighted taut-string solve for validated input with n >= 2.

    Two numerical guards keep the worst case far inside the 1e-6 tolerance:

    * the signal is shifted by ``y[0]``, which is an exact symmetry of the
      functional (``x -> x + c``), and removes the drift of the cumulative
      sums when the level is large;
    * the running sums use Kahan compensation, so their error stays
      proportional to the current sum instead of growing with ``n``.
    """
    n = len(y)
    base = float(y[0])

    tau = array("d", [0.0]) * (n + 1)   # cumulative weight
    cum = array("d", [0.0]) * (n + 1)   # cumulative weighted signal (shifted)
    low = array("d", [0.0]) * (n + 1)   # lower tube edge
    up = array("d", [0.0]) * (n + 1)    # upper tube edge

    t = 0.0; ct = 0.0
    s = 0.0; cs = 0.0
    for i in range(n):
        wi = w[i]
        # Kahan-compensated accumulation of tau.
        q = wi - ct
        t_new = t + q
        ct = (t_new - t) - q
        t = t_new
        # Kahan-compensated accumulation of the shifted weighted signal.
        q = wi * (y[i] - base) - cs
        s_new = s + q
        cs = (s_new - s) - q
        s = s_new
        tau[i + 1] = t
        cum[i + 1] = s
    for i in range(1, n):
        pi = p[i - 1]
        si = cum[i]
        low[i] = si - pi
        up[i] = si + pi
    low[n] = cum[n]
    up[n] = cum[n]

    # Lower and upper hull stacks, plus the accepted knots of the string.
    l_idx = array("i", [0]) * (n + 1)
    l_slp = array("d", [0.0]) * (n + 1)
    u_idx = array("i", [0]) * (n + 1)
    u_slp = array("d", [0.0]) * (n + 1)
    k_idx = array("i", [0]) * (n + 1)
    k_val = array("d", [0.0]) * (n + 1)

    s_low = c_low = 0
    s_up = c_up = 0
    c = 0
    l_slp[0] = _INF
    u_slp[0] = -_INF

    for i in range(1, n + 1):
        dt_prev = tau[i] - tau[i - 1]
        li = low[i]
        ui = up[i]

        # Push i onto the lower hull, popping non-convex predecessors.
        c_low += 1
        l_idx[c_low] = i
        l_slp[c_low] = (li - low[i - 1]) / dt_prev
        while c_low > s_low + 1:
            if l_slp[c_low - 1] <= l_slp[c_low]:
                c_low -= 1
                l_idx[c_low] = i
                if c_low > s_low + 1:
                    q = l_idx[c_low - 1]
                    l_slp[c_low] = (li - low[q]) / (tau[i] - tau[q])
                else:
                    q = k_idx[c]
                    l_slp[c_low] = (li - k_val[c]) / (tau[i] - tau[q])
            else:
                break

        # Push i onto the upper hull, popping non-convex predecessors.
        c_up += 1
        u_idx[c_up] = i
        u_slp[c_up] = (ui - up[i - 1]) / dt_prev
        while c_up > s_up + 1:
            if u_slp[c_up - 1] >= u_slp[c_up]:
                c_up -= 1
                u_idx[c_up] = i
                if c_up > s_up + 1:
                    q = u_idx[c_up - 1]
                    u_slp[c_up] = (ui - up[q]) / (tau[i] - tau[q])
                else:
                    q = k_idx[c]
                    u_slp[c_up] = (ui - k_val[c]) / (tau[i] - tau[q])
            else:
                break

        # The hulls have crossed: pin a new knot and restart from it.
        while c_low == s_low + 1 and c_up > s_up + 1 and l_slp[c_low] >= u_slp[s_up + 1]:
            s_up += 1
            c += 1
            q = u_idx[s_up]
            k_idx[c] = q
            k_val[c] = up[q]
            l_idx[s_low] = q
            l_slp[c_low] = (li - k_val[c]) / (tau[i] - tau[q])

        while c_up == s_up + 1 and c_low > s_low + 1 and u_slp[c_up] <= l_slp[s_low + 1]:
            s_low += 1
            c += 1
            q = l_idx[s_low]
            k_idx[c] = q
            k_val[c] = low[q]
            u_idx[s_up] = q
            u_slp[c_up] = (ui - k_val[c]) / (tau[i] - tau[q])

    # The lower hull that is left over traces the string to the endpoint.
    for q in range(s_low + 1, c_low + 1):
        c += 1
        k = l_idx[q]
        k_idx[c] = k
        k_val[c] = low[k]

    # The slope of the string between consecutive knots is the signal.
    # Knot indices are strictly increasing: every accepted knot is a node
    # already popped from a stack, and an accepted knot at index i clears the
    # stacks down to the segment starting at i.
    x = [0.0] * n
    prev_i = k_idx[0]
    prev_v = k_val[0]
    for q in range(1, c + 1):
        i = k_idx[q]
        slope = (k_val[q] - prev_v) / (tau[i] - tau[prev_i])
        level = slope + base
        for j in range(prev_i, i):
            x[j] = level
        prev_i = i
        prev_v = k_val[q]
    return x


if __name__ == "__main__":  # pragma: no cover
    print(denoise([0, 4], [1, 3], [1]))
