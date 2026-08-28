"""Curvature metrics. Pure functions, numpy, no I/O. Unit-tested in test_metrics.py against the
analytic nulls before running on real data. All angles returned in RADIANS; use deg() to convert.
No em-dashes. Keep fp64 for the arccos step since the cosine argument near +/-1 is sensitive.
"""
from __future__ import annotations
import numpy as np

TWO_PI_3 = 2.0 * np.pi / 3.0      # 120 deg, iid null
HALF_PI = np.pi / 2.0            # 90 deg, random-walk null


def deg(x):
    return np.degrees(x)


def difference_vectors(X: np.ndarray) -> np.ndarray:
    """v_k = x_{k+1} - x_k. X is [n, d]; returns [n-1, d]."""
    return np.diff(np.asarray(X, dtype=np.float64), axis=0)


def pointwise_curvature(X: np.ndarray) -> np.ndarray:
    """c_k = arccos(<v_{k+1}, v_k> / (||v_{k+1}|| ||v_k||)), radians. X [n, d] -> [n-2].
    Cosine clipped to [-1, 1]. Zero-norm difference vectors (repeated identical states) -> NaN,
    propagated rather than substituted."""
    v = difference_vectors(X)
    if v.shape[0] < 2:
        return np.empty(0)
    a, b = v[:-1], v[1:]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.einsum("ij,ij->i", a, b) / denom
    cos = np.clip(cos, -1.0, 1.0)
    c = np.arccos(cos)
    c[denom == 0] = np.nan
    return c


def contextual_curvature(c: np.ndarray, window: int = 3):
    """Backward windowed curvature. For window 3: C_k = (1/3) sum_{i=k-4}^{k-2} c_i, defined for
    k >= 6 (1-indexed k, matching the trajectory index). General window w uses the w pointwise
    curvatures ending at c_{k-2}, first valid k = w + 3. Returns (k_1indexed, C). NaN propagates."""
    c = np.asarray(c, dtype=np.float64)
    m = c.shape[0]                       # m = n - 2 pointwise curvatures, c[0] == c_1
    k_min = window + 3                   # 6 for window 3
    k_max = m + 2                        # need hi = k-2 <= m
    ks, Cs = [], []
    for k in range(k_min, k_max + 1):
        lo, hi = k - window - 1, k - 2   # 1-indexed inclusive window [c_lo, c_hi], w terms
        seg = c[lo - 1:hi]               # 0-indexed slice
        Cs.append(seg.mean())            # plain mean -> NaN if any term is NaN
        ks.append(k)
    return np.asarray(ks, dtype=int), np.asarray(Cs, dtype=float)


def step_norms(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(difference_vectors(X), axis=1)


def path_length(X: np.ndarray) -> float:
    return float(step_norms(X).sum())


def chord_length(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    return float(np.linalg.norm(X[-1] - X[0]))


def straightness(X: np.ndarray) -> float:
    """S = chord / path in (0, 1]; 1 is a straight path. NaN if path length is 0."""
    L = path_length(X)
    return float(chord_length(X) / L) if L > 0 else np.nan


def trajectory_summary(X: np.ndarray) -> dict:
    """Cheap per-trajectory statistics computed alongside curvature."""
    c = pointwise_curvature(X)
    return {
        "n_points": int(np.asarray(X).shape[0]),
        "mean_curv_rad": float(np.nanmean(c)) if c.size else np.nan,
        "median_curv_rad": float(np.nanmedian(c)) if c.size else np.nan,
        "path_length": path_length(X),
        "chord_length": chord_length(X),
        "straightness": straightness(X),
        "mean_step_norm": float(step_norms(X).mean()) if X.shape[0] > 1 else np.nan,
    }


# Two named axes. Same computation, different axis. NEVER pool them in outputs.
def position_curvature(X_tokens: np.ndarray) -> np.ndarray:
    """Curvature over token positions at a fixed layer. X_tokens [n_tokens, d]. Case A only."""
    return pointwise_curvature(X_tokens)


def depth_curvature(X_layers: np.ndarray) -> np.ndarray:
    """Curvature over layers at a fixed token/pooling. X_layers [n_layers, d]. Case A or B."""
    return pointwise_curvature(X_layers)


def twonn(X: np.ndarray, fractions=(0.25, 0.5, 1.0), seed: int = 0) -> dict:
    """TwoNN intrinsic dimension (Facco et al. 2017), reported at several subsample fractions to
    check the plateau. Never returns a single number. X [N, d]."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    out = {}
    for frac in fractions:
        n = max(10, int(round(frac * N)))
        idx = rng.choice(N, size=min(n, N), replace=False)
        Xs = X[idx]
        nn = NearestNeighbors(n_neighbors=3).fit(Xs)     # self + 2 neighbors
        dist, _ = nn.kneighbors(Xs)
        r1, r2 = dist[:, 1], dist[:, 2]
        good = (r1 > 0)
        mu = r2[good] / r1[good]
        mu = mu[mu > 1]                                   # drop degenerate mu == 1
        mu_sorted = np.sort(mu)
        F = np.arange(1, mu_sorted.size + 1) / mu_sorted.size
        keep = F < 1.0
        x = np.log(mu_sorted[keep])
        y = -np.log(1.0 - F[keep])
        d_est = float(np.dot(x, y) / np.dot(x, x)) if x.size else np.nan   # slope through origin
        out[f"frac_{frac}"] = {"n": int(Xs.shape[0]), "id": d_est}
    return out
