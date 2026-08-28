"""G1 gate: analytic nulls and guards for metrics.py. Runnable without pytest:
    python test_metrics.py
Asserts curvature reproduces the iid (120 deg) and random-walk (90 deg) nulls to within 2 deg,
the contextual-curvature first valid index is 6 (window 3), NaN guards fire, and clipping holds.
No em-dashes.
"""
from __future__ import annotations
import numpy as np
import metrics as M

TOL_DEG = 2.0
RNG = np.random.default_rng(0)


def _mean_curv_deg(X):
    return float(np.degrees(np.nanmean(M.pointwise_curvature(X))))


def test_iid_null():
    # x_i iid N(0, I). E[cos] -> -1/2, curvature -> 120 deg. Needs many points and dims.
    X = RNG.standard_normal((6000, 512))
    got = _mean_curv_deg(X)
    assert abs(got - 120.0) < TOL_DEG, f"iid null: got {got:.2f} deg, expected 120"
    return got


def test_random_walk_null():
    # x_{k+1} = x_k + eta. Difference vectors independent, E[cos] = 0, curvature -> 90 deg.
    steps = RNG.standard_normal((6000, 512))
    X = np.cumsum(steps, axis=0)
    got = _mean_curv_deg(X)
    assert abs(got - 90.0) < TOL_DEG, f"random-walk null: got {got:.2f} deg, expected 90"
    return got


def test_contextual_first_index():
    c = np.arange(1.0, 12.0)          # 11 pointwise curvatures
    ks, Cs = M.contextual_curvature(c, window=3)
    assert ks[0] == 6, f"first valid contextual index: got {ks[0]}, expected 6"
    # C_6 = mean(c_2, c_3, c_4) = mean(2,3,4) = 3.0  (1-indexed c)
    assert abs(Cs[0] - 3.0) < 1e-9, f"C_6: got {Cs[0]}, expected 3.0"
    return int(ks[0])


def test_nan_guard():
    # A repeated identical state gives a zero-norm difference vector -> NaN at that junction.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    c = M.pointwise_curvature(X)
    assert np.isnan(c).any(), "zero-norm step should propagate NaN"
    return int(np.isnan(c).sum())


def test_clip_and_straightness():
    # Straight line: curvature ~ 0, straightness == 1.
    X = np.stack([np.linspace(0, 10, 20), np.zeros(20)], axis=1)
    c = M.pointwise_curvature(X)
    assert np.nanmax(np.degrees(c)) < 1e-3, "straight line should have ~0 curvature"
    assert abs(M.straightness(X) - 1.0) < 1e-9, "straight line straightness should be 1"
    # Sharp reversal: curvature ~ 180 deg, no NaN from clipping.
    Z = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    cz = M.pointwise_curvature(Z)
    assert abs(np.degrees(cz[0]) - 180.0) < 1e-6 and not np.isnan(cz[0]), "reversal should be 180 deg"
    return float(M.straightness(X))


if __name__ == "__main__":
    tests = [test_iid_null, test_random_walk_null, test_contextual_first_index,
             test_nan_guard, test_clip_and_straightness]
    ok = True
    for t in tests:
        try:
            r = t()
            print(f"PASS  {t.__name__:28s} -> {r}")
        except AssertionError as e:
            ok = False; print(f"FAIL  {t.__name__:28s} -> {e}")
    print("\nG1", "PASS (nulls within 2 deg)" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
