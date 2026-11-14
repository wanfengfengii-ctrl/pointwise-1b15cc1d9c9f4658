#!/usr/bin/env python3
"""Command line front end for the weighted 1-D TV denoiser.

Subcommands
-----------
verify   run the unit-test suite and an independent optimality audit
         (exact rational active-set enumeration on tiny inputs plus KKT
         residuals on medium inputs)
demo     denoise a synthetic long instrument trace and report how well the
         step is preserved
bench    measure run time and peak memory on the doubling series
         25k -> 50k -> 100k -> 200k samples, each solve in a fresh child
         process so the peaks do not contaminate each other

Only the Python standard library is used.
"""

import json
import os
import random
import resource
import subprocess
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "tests")
for _p in (HERE, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tv import denoise

SIZES = (25_000, 50_000, 100_000, 200_000)


# --------------------------------------------------------------------------
# signal generators (deterministic, standard library only)
# --------------------------------------------------------------------------

# Instrument trace: piecewise-constant truth with two abrupt steps, slow
# linear drift and Gaussian noise.  Step positions are index based (80 and
# 180) for small n, and fractions of n for long traces.
def _step_positions(n):
    if n <= 400:
        return (min(80, n), min(180, n))
    return (n // 4, 9 * n // 10)


def instrument_trace(n, seed=2024):
    """Long noisy trace of an instrument reading with abrupt step changes.

    Returns (y, truth, weights, penalties) with per-point weights of 1 and
    penalties of 2.
    """
    rng = random.Random(seed)
    s1, s2 = _step_positions(n)
    levels = (-2.5, 3.5, 0.5)
    truth = [levels[0]] * s1 + [levels[1]] * (s2 - s1) + [levels[2]] * (n - s2)
    y = []
    for i, v in enumerate(truth):
        drift = 0.25 * (i / max(n - 1, 1))
        y.append(v + drift + rng.gauss(0.0, 0.35))
    weights = [1.0] * n
    penalties = [2.0] * (n - 1) if n > 1 else []
    return y, truth, weights, penalties


def random_trace(n, seed=7):
    rng = random.Random(seed)
    return (
        [rng.uniform(-1e4, 1e4) for _ in range(n)],
        [rng.uniform(0.1, 10.0) for _ in range(n)],
        [rng.uniform(0.0, 200.0) for _ in range(n - 1)],
    )


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

def _exact_audit():
    from exact_reference import solve_exact, objective
    rng = random.Random(101)
    worst = 0.0
    cases = 0
    for _ in range(120):
        n = rng.randint(1, 6)
        y = [rng.choice([-1e4, -1.0, 0.0, 0.5, 1e4])
             if rng.random() < 0.6 else rng.uniform(-40.0, 40.0)
             for _ in range(n)]
        w = [rng.choice([0.1, 1.0, 10.0])
             if rng.random() < 0.6 else rng.uniform(0.1, 10.0)
             for _ in range(n)]
        p = [rng.choice([0.0, 1e4])
             if rng.random() < 0.6 else rng.uniform(0.0, 15.0)
             for _ in range(n - 1)]
        got = denoise(y, w, p)
        want = solve_exact(y, w, p)
        err = max(float(abs(Fraction(a) - b)) for a, b in zip(got, want))
        gap = float(objective(y, w, p, got) - objective(y, w, p, want))
        worst = max(worst, err)
        if err > 1e-6 * (1.0 + max(abs(v) for v in y)) or gap > 1e-6:
            raise SystemExit(
                "exact audit failed at n=%d: err=%r gap=%r\n y=%r w=%r p=%r"
                % (n, err, gap, y, w, p)
            )
        cases += 1
    return cases, worst


def _kkt_audit():
    from exact_reference import kkt_violation
    rng = random.Random(202)
    worst = 0.0
    cases = 0
    naive = 0.0
    for n, count in ((50, 60), (500, 12), (5000, 2)):
        for _ in range(count):
            y = [rng.uniform(-1e4, 1e4) for _ in range(n)]
            w = [rng.uniform(0.1, 10.0) for _ in range(n)]
            p = [rng.choice([0.0, 1e4]) if rng.random() < 0.2
                 else rng.uniform(0.0, 1000.0) for _ in range(n - 1)]
            x = denoise(y, w, p)
            viol, where = kkt_violation(y, w, p, x)
            naive = max(naive, viol / (1.0 + max(abs(v) for v in y)))
            if viol > 1e-6 * (1.0 + max(abs(v) for v in y)):
                raise SystemExit(
                    "KKT audit failed at n=%d (%s): violation=%r" % (n, where, viol)
                )
            worst = max(worst, viol)
            cases += 1
    return cases, worst, naive


def cmd_verify(_args):
    import unittest

    print("== tv.denoise verification ==")
    print("documented examples")
    ex = denoise([0, 4], [1, 3], [1])
    assert denoise([], [], []) == []
    assert denoise([1], [1], []) == [1.0]
    assert denoise([0, 4], [1, 3], [0]) == [0.0, 4.0]
    assert abs(ex[0] - 1.0) < 1e-15 and abs(ex[1] - 11.0 / 3.0) < 1e-15
    print("   [0,4],[1,3],[1]  -> %r  (expected [1, 11/3])" % (ex,))
    print("   [0,4],[1,3],[0]  -> %r  (expected [0, 4])"
          % (denoise([0, 4], [1, 3], [0]),))
    print("   [1],[1],[]       -> %r" % (denoise([1], [1], []),))
    print("   [],[],[]         -> %r" % (denoise([], [], []),))

    print("independent exact audit (rational active-set enumeration)")
    cases, worst = _exact_audit()
    print("   %d tiny cases, max |x - x_exact| = %.3e" % (cases, worst))

    print("KKT residual audit (first-order optimality, floats)")
    cases, worst, relative = _kkt_audit()
    print("   %d cases up to n=5000, max violation = %.3e (%.3e relative)"
          % (cases, worst, relative))

    print("unit test suite")
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(HERE, "tests"), pattern="test_*.py",
                            top_level_dir=HERE)
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("VERIFY OK")


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------

def _rmse(a, b):
    return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5


def _print_series(name, values, stride):
    shown = values[::stride]
    print("   %s: %s" % (name, " ".join("%7.3f" % v for v in shown)))


def cmd_demo(args):
    n = 200_000 if "--full" in args else 2000
    y, truth, w, p = instrument_trace(n)
    t0 = time.perf_counter()
    x = denoise(y, w, p)
    elapsed = time.perf_counter() - t0

    print("== denoising demo (instrument trace, n=%d) ==" % n)
    print("model: 0.5*sum w_i (x_i-y_i)^2 + sum p_i |x_{i+1}-x_i|,"
          " w_i=1, p_i=2")
    print("noisy RMSE vs truth  = %.4f" % _rmse(y, truth))
    print("denoised RMSE        = %.4f" % _rmse(x, truth))
    print("TV of noisy signal   = %.3f"
          % sum(abs(y[i + 1] - y[i]) for i in range(n - 1)))
    print("TV of denoised signal= %.3f"
          % sum(abs(x[i + 1] - x[i]) for i in range(n - 1)))
    print("solve time           = %.3f s" % elapsed)

    # Step preservation: average the plateau on each side, then compare the
    # recovered jump with the jump in the ground truth.
    s1, s2 = _step_positions(n)
    print("step preservation (plateau means 40 points either side)")
    for centre in (s1, s2):
        if centre < 45 or centre + 45 > n:
            continue
        before = sum(x[centre - 45:centre - 5]) / 40.0
        after = sum(x[centre + 5:centre + 45]) / 40.0
        t_before = truth[centre - 25]
        t_after = truth[centre + 25]
        print("   step at %6d: denoised %.3f -> %.3f (jump %+.3f, truth %+.3f,"
              " error %.1f%%)"
              % (centre, before, after, after - before, t_after - t_before,
                 100.0 * abs((after - before) - (t_after - t_before))
                 / abs(t_after - t_before)))
    if n >= 300:
        stride = max(n // 10, 1)
        print("sample values (every %d-th point)" % stride)
        _print_series("noisy   ", y, stride)
        _print_series("denoised", x, stride)

    print("DEMO OK")


# --------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------

def _peak_kib():
    """Process peak RSS in KiB from /proc (Linux) with a getrusage fallback."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


_KINDS = {
    "random": lambda n, rng: [rng.uniform(-1e4, 1e4) for _ in range(n)],
    "step": lambda n, rng: [-1e4 if i < n // 2 else 1e4 for i in range(n)],
    "constant": lambda n, rng: [1e4] * n,
    "alternating": lambda n, rng: [1e4 if i % 2 else -1e4 for i in range(n)],
    "spikes": lambda n, rng: [1e4 if i % 997 == 0 else 0.0 for i in range(n)],
}


def _bench_child(n, kind):
    rng = random.Random(n)
    y = _KINDS[kind](n, rng)
    w = [5.0] * n
    p = [0.5] * (n - 1)
    base = _peak_kib()
    t0 = time.perf_counter()
    x = denoise(y, w, p)
    elapsed = time.perf_counter() - t0
    peak = _peak_kib()
    checksum = x[0] + x[-1] + len(x)
    print(json.dumps({
        "n": n,
        "kind": kind,
        "seconds": elapsed,
        "peak_kib": peak,
        "base_kib": base,
        "delta_kib": max(peak - base, 0),
        "checksum": checksum,
    }))


def _run_child(n, kind):
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "bench-child",
         str(n), kind],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _scaling_table():
    print("   %10s %5s %12s %10s %12s %12s" %
          ("n", "kind", "time", "vs prev", "peak delta", "per sample"))
    rows = []
    by_kind = {}
    kinds = ("random", "step", "constant", "alternating", "spikes")
    for n in SIZES:
        for kind in kinds:
            row = _run_child(n, kind)
            prev = by_kind.get(kind)
            ratio = row["seconds"] / prev if prev else None
            print("   %10d %5s %11.3fs %10s %10.1f MiB %10.1f B"
                  % (n, kind, row["seconds"],
                     ("%.2fx" % ratio) if ratio else "-",
                     row["delta_kib"] / 1024.0,
                     row["delta_kib"] * 1024.0 / n))
            by_kind[kind] = row["seconds"]
            rows.append(row)
    return rows


def cmd_bench(args):
    print("== doubling benchmark (fresh process per solve) ==")
    print("signal: y in [-1e4,1e4], weights all 5, penalties all 0.5")
    rows = _scaling_table()
    worst = max(rows, key=lambda r: r["seconds"])
    print("worst case: n=%d kind=%s time=%.3fs peak delta=%.1f MiB"
          % (worst["n"], worst["kind"], worst["seconds"],
             worst["delta_kib"] / 1024.0))
    print("BENCH OK")


# --------------------------------------------------------------------------

def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "verify":
        cmd_verify(rest)
    elif cmd == "demo":
        cmd_demo(rest)
    elif cmd == "bench":
        cmd_bench(rest)
    elif cmd == "bench-child":
        _bench_child(int(rest[0]), rest[1])
    else:
        print("unknown subcommand %r" % cmd, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
