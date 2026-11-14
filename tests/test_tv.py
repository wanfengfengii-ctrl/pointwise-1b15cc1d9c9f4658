"""Tests for tv.denoise.

Run with either of

    python3 tests/test_tv.py
    python3 -m unittest discover -s tests -t .

All tests use only the standard library.  Optimality is certified two
independent ways: exact rational active-set enumeration on tiny inputs, and
first-order KKT residual checks on larger inputs.
"""

import os
import random
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from tv import denoise
from exact_reference import solve_exact, objective, kkt_violation
from fractions import Fraction


def tolerance(y):
    return 1e-6 * (1.0 + max((abs(v) for v in y), default=0.0))


class TestExamples(unittest.TestCase):
    def test_documented_examples(self):
        self.assertEqual(denoise([], [], []), [])
        self.assertEqual(denoise([1], [1], []), [1.0])
        self.assertEqual(denoise([0, 4], [1, 3], [0]), [0.0, 4.0])
        got = denoise([0, 4], [1, 3], [1])
        self.assertAlmostEqual(got[0], 1.0, places=15)
        self.assertAlmostEqual(got[1], 11.0 / 3.0, places=15)

    def test_returns_floats(self):
        x = denoise([1, 5, 2], [1, 2, 3], [4, 5])
        self.assertEqual(len(x), 3)
        for v in x:
            self.assertIsInstance(v, float)

    def test_zero_penalty_is_identity(self):
        r = random.Random(11)
        y = [r.uniform(-1e4, 1e4) for _ in range(200)]
        w = [r.uniform(0.1, 10) for _ in range(200)]
        x = denoise(y, w, [0.0] * 199)
        for a, b in zip(x, y):
            self.assertAlmostEqual(a, b, delta=tolerance(y))

    def test_large_penalty_collapses_to_weighted_mean(self):
        y = [-4.0, 7.0, 2.0, 9.0]
        w = [1.0, 2.0, 3.0, 4.0]
        x = denoise(y, w, [1e4, 1e4, 1e4])
        mean = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
        for v in x:
            self.assertAlmostEqual(v, mean, places=12)


class TestInvariants(unittest.TestCase):
    def test_constant_signal_stays_constant(self):
        for level in (-1e4, -3.5, 0.0, 42.0, 1e4):
            x = denoise([level] * 1000, [1.0] * 1000, [10000.0] * 999)
            self.assertEqual(set(x), {float(level)})

    def test_shift_invariance(self):
        r = random.Random(2)
        y = [r.uniform(-1e4, 1e4) for _ in range(120)]
        w = [r.uniform(0.1, 10) for _ in range(120)]
        p = [r.uniform(0, 500) for _ in range(119)]
        x = denoise(y, w, p)
        for c in (-1e4, 1.5, 12345.0):
            if max(abs(v + c) for v in y) > 1e4:
                continue
            shifted = denoise([v + c for v in y], w, p)
            for a, b in zip(shifted, x):
                self.assertAlmostEqual(a, b + c, delta=tolerance(y))

    def test_common_weight_scale_is_invariant(self):
        r = random.Random(3)
        y = [r.uniform(-1, 1) for _ in range(50)]
        w = [r.uniform(0.1, 10) for _ in range(50)]
        p = [r.uniform(0, 20) for _ in range(49)]
        x = denoise(y, w, p)
        x2 = denoise(y, [0.5 * v for v in w], [0.5 * v for v in p])
        for a, b in zip(x, x2):
            self.assertAlmostEqual(a, b, delta=tolerance(y))

    def test_reversal(self):
        r = random.Random(4)
        y = [r.uniform(-1e4, 1e4) for _ in range(60)]
        w = [r.uniform(0.1, 10) for _ in range(60)]
        p = [r.uniform(0, 300) for _ in range(59)]
        x = denoise(y, w, p)
        xr = denoise(y[::-1], w[::-1], p[::-1])
        for a, b in zip(x, xr[::-1]):
            self.assertAlmostEqual(a, b, delta=tolerance(y))

    def test_zero_penalty_decouples_the_edge(self):
        # p = 0 makes that edge free: the two sides are independent problems.
        x = denoise([0.0, 5.0, -3.0], [1.0, 1.0, 1.0], [1e4, 0.0])
        want = solve_exact([0.0, 5.0, -3.0], [1.0, 1.0, 1.0], [1e4, 0.0])
        for a, b in zip(x, want):
            self.assertLessEqual(abs(Fraction(a) - b), Fraction(tolerance([0, 5])))
        x2 = denoise([0.0, 5.0, -3.0], [1.0, 1.0, 1.0], [0.0, 1e4])
        self.assertAlmostEqual(x2[0], 0.0, places=12)
        self.assertAlmostEqual(x2[1], 1.0, places=12)
        self.assertAlmostEqual(x2[2], 1.0, places=12)

    def test_noise_is_reduced(self):
        r = random.Random(5)
        truth = [0.0 if i < 100 else 10.0 for i in range(200)]
        y = [v + r.gauss(0, 1.5) for v in truth]
        x = denoise(y, [1.0] * 200, [30.0] * 199)
        rms_before = (sum((a - b) ** 2 for a, b in zip(y, truth)) / 200.0) ** 0.5
        rms_after = (sum((a - b) ** 2 for a, b in zip(x, truth)) / 200.0) ** 0.5
        self.assertLess(rms_after, rms_before)
        self.assertLess(abs(x[50] - 0.0), 1.0)
        self.assertLess(abs(x[150] - 10.0), 1.0)


class TestValidation(unittest.TestCase):
    def assertBad(self, y, w, p):
        with self.assertRaises(ValueError):
            denoise(y, w, p)

    def test_lengths(self):
        self.assertBad([1, 2], [1], [])
        self.assertBad([1, 2], [1, 1, 1], [1])
        self.assertBad([1, 2], [1, 1], [1, 2])
        self.assertBad([], [], [1])
        self.assertBad([1], [1], [1])
        self.assertBad([1, 2, 3], [1, 2, 3], [1])

    def test_non_lists(self):
        self.assertBad((1, 2), [1, 1], [1])
        self.assertBad([1, 2], (1, 1), [1])
        self.assertBad([1, 2], [1, 1], (1,))
        self.assertBad("ab", [1, 1], [1])
        self.assertBad(None, [1, 1], [1])

    def test_bad_values(self):
        self.assertBad([True, False], [1, 1], [1])
        self.assertBad([1, 2], [True, 1], [1])
        self.assertBad([1, 2], [1, 1], [True])
        self.assertBad([1, 2], [1, 1], [None])
        self.assertBad([1, 2], [1, 1], ["2"])
        self.assertBad([1, 2], [1, 1], [float("nan")])
        self.assertBad([1, 2], [1, 1], [float("inf")])
        self.assertBad([float("nan"), 2], [1, 1], [1])
        self.assertBad([float("inf"), 2], [1, 1], [1])
        self.assertBad([1e9, 2], [1, 1], [1])
        self.assertBad([1, 2], [0.05, 1], [1])
        self.assertBad([1, 2], [11, 1], [1])
        self.assertBad([1, 2], [1, 1], [-1])
        self.assertBad([1, 2], [1, 1], [20000])

    def test_too_long(self):
        n = 200_001
        self.assertBad([0.0] * n, [1.0] * n, [0.0] * (n - 1))

    def test_accepts_boundaries(self):
        x = denoise([-1e4, 1e4], [0.1, 10], [1e4])
        self.assertEqual(len(x), 2)
        x = denoise([0, 0], [1, 1], [0])
        self.assertEqual(x, [0.0, 0.0])

    def test_input_not_mutated(self):
        y = [0, 4]
        w = [1, 3]
        p = [1]
        cp = (y[:], w[:], p[:])
        denoise(y, w, p)
        self.assertEqual((y, w, p), cp)
        self.assertEqual([type(v) for v in y], [int, int])

    def test_input_not_mutated_long(self):
        r = random.Random(9)
        y = [r.uniform(-1e4, 1e4) for _ in range(5000)]
        w = [r.uniform(0.1, 10) for _ in range(5000)]
        p = [r.uniform(0, 100) for _ in range(4999)]
        cp = (y[:], w[:], p[:])
        denoise(y, w, p)
        self.assertEqual((y, w, p), cp)


class TestOptimality(unittest.TestCase):
    def test_exact_small_cases(self):
        r = random.Random(6)
        for trial in range(60):
            n = r.randint(1, 6)
            y = [r.choice([-1e4, -1.0, 0.0, 0.25, 1.0, 1e4])
                 if r.random() < 0.5 else r.uniform(-50, 50) for _ in range(n)]
            w = [r.choice([0.1, 1.0, 10.0]) if r.random() < 0.5
                 else r.uniform(0.1, 10) for _ in range(n)]
            p = [r.choice([0.0, 1e4]) if r.random() < 0.5
                 else r.uniform(0, 20) for _ in range(max(n - 1, 0))]
            got = denoise(y, w, p)
            want = solve_exact(y, w, p)
            for a, b in zip(got, want):
                self.assertLessEqual(abs(Fraction(a) - b), Fraction(1, 10 ** 9),
                                     "trial %d n=%d" % (trial, n))
            self.assertLessEqual(
                objective(y, w, p, got) - objective(y, w, p, want),
                Fraction(tolerance(y)) * 10,
            )

    def test_accuracy_bound_vs_exact(self):
        r = random.Random(7)
        worst = 0.0
        for _ in range(40):
            n = r.randint(1, 7)
            y = [r.uniform(-1e4, 1e4) for _ in range(n)]
            w = [r.uniform(0.1, 10) for _ in range(n)]
            p = [r.uniform(0, 10000) for _ in range(n - 1)]
            got = denoise(y, w, p)
            want = solve_exact(y, w, p)
            err = max(abs(Fraction(a) - b) for a, b in zip(got, want))
            worst = max(worst, float(err))
            self.assertLessEqual(float(err), tolerance(y))
        self.assertGreaterEqual(worst, 0.0)

    def test_kkt_small(self):
        r = random.Random(8)
        for _ in range(80):
            n = r.randint(2, 40)
            y = [r.uniform(-1e4, 1e4) for _ in range(n)]
            w = [r.uniform(0.1, 10) for _ in range(n)]
            p = [r.choice([0.0, 1e4]) if r.random() < 0.3
                 else r.uniform(0, 1000) for _ in range(n - 1)]
            x = denoise(y, w, p)
            viol, where = kkt_violation(y, w, p, x)
            self.assertLessEqual(viol, 1e-6 * (1 + max(abs(v) for v in y)),
                                 where)

    def test_kkt_medium(self):
        r = random.Random(9)
        n = 4000
        y = [r.uniform(-1e4, 1e4) for _ in range(n)]
        w = [r.uniform(0.1, 10) for _ in range(n)]
        p = [r.uniform(0, 200) for _ in range(n - 1)]
        x = denoise(y, w, p)
        viol, where = kkt_violation(y, w, p, x)
        self.assertLessEqual(viol, 1e-6 * (1 + max(abs(v) for v in y)), where)

    def test_exact_at_extremes(self):
        cases = [
            ([-1e4, 1e4], [0.1, 0.1], [0.0]),
            ([1e4, -1e4], [10.0, 10.0], [1e4]),
            ([-1e4, -1e4], [0.1, 10.0], [0.0]),
            ([-1e4, 1e4, -1e4], [0.1, 10.0, 0.1], [0.0, 1e4]),
            ([1e4, 1e4, 1e4], [10.0, 0.1, 10.0], [0.0, 0.0]),
        ]
        for y, w, p in cases:
            got = denoise(y, w, p)
            want = solve_exact(y, w, p)
            for a, b in zip(got, want):
                self.assertLessEqual(abs(Fraction(a) - b),
                                     Fraction(tolerance(y)))


class TestScale(unittest.TestCase):
    def test_200k_worst_case_patterns(self):
        n = 200_000
        r = random.Random(10)
        patterns = [
            [1e4] * n,
            [-1e4 if i < n // 2 else 1e4 for i in range(n)],
            [1e4 if i % 2 else -1e4 for i in range(n)],
            [1e4 if i % 997 == 0 else 0.0 for i in range(n)],
            [r.uniform(-1e4, 1e4) for _ in range(n)],
        ]
        for y in patterns:
            t0 = time.perf_counter()
            x = denoise(y, [0.1] * n, [0.0] * (n - 1))
            elapsed = time.perf_counter() - t0
            self.assertEqual(len(x), n)
            self.assertLess(elapsed, 20.0, "unexpected superlinear blow-up")
            viol, where = kkt_violation(y, [0.1] * n, [0.0] * (n - 1), x)
            self.assertLessEqual(viol, 1e-5 * (1 + max(abs(v) for v in y)),
                                 where)


if __name__ == "__main__":
    unittest.main(verbosity=2)
