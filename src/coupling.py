import numpy as np
import skdim

THRESHOLD = 0.1
MINIMUM_POINTS = 200


def _twonn(data: np.ndarray) -> float:
    """Fit TwoNN and return intrinsic dimensionality estimate."""
    return skdim.id.TwoNN().fit(data).dimension_


def compute_marginals(noise_read: np.ndarray,
                      geometry_read: np.ndarray,
                      clock_read: np.ndarray) -> dict:
    """
    Compute marginal dimensionalities once.
    Each read must be shape (n, 1) with n >= MINIMUM_POINTS.
    """
    return {
        "d_noise":    _twonn(noise_read),
        "d_geometry": _twonn(geometry_read),
        "d_clock":    _twonn(clock_read),
    }


def compute_gaps(noise_read: np.ndarray,
                 geometry_read: np.ndarray,
                 clock_read: np.ndarray,
                 marginals: dict) -> dict:
    """
    Compute four coupling gaps from reads and pre-computed marginals.
    Gap = joint dimensionality - sum of marginal dimensionalities.
    Positive and large = coupled = shared physical cause.
    Near zero = decoupled = causal chain absent or broken.
    """
    d_n = marginals["d_noise"]
    d_g = marginals["d_geometry"]
    d_c = marginals["d_clock"]

    joint_ng = np.concatenate([noise_read, geometry_read], axis=1)
    joint_nc = np.concatenate([noise_read, clock_read], axis=1)
    joint_gc = np.concatenate([geometry_read, clock_read], axis=1)
    joint_all = np.concatenate([noise_read, geometry_read, clock_read], axis=1)

    return {
        "noise_geometry_gap": _twonn(joint_ng)  - d_n - d_g,
        "noise_clock_gap":    _twonn(joint_nc)  - d_n - d_c,
        "geometry_clock_gap": _twonn(joint_gc)  - d_g - d_c,
        "joint_all_gap":      _twonn(joint_all) - d_n - d_g - d_c,
    }


def verdict(gaps: dict) -> str:
    """
    Three-sided verdict from four gap numbers.
    Returns 'real', 'synthetic', or 'manipulated'.
    """
    values = list(gaps.values())
    if all(v >= THRESHOLD for v in values):
        return "real"
    if all(v < THRESHOLD for v in values):
        return "synthetic"
    return "manipulated"


def which_pairs_broke(gaps: dict) -> list:
    return [pair for pair, val in gaps.items() if val < THRESHOLD]


def surviving_pairs(gaps: dict) -> list:
    return [pair for pair, val in gaps.items() if val >= THRESHOLD]
