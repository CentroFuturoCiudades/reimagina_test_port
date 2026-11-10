"""
composite.py — the math of composite indicators.

Indicators normalise; composites combine their (already-normalised) children
with the weighted Mazziotta-Pareto formula. The tree that says which
indicators feed which composite is ``composite_config.yaml``;
``composite_metric`` turns it into one Lyra node per composite.

    soft_threshold          distance -> 0-1 access score
    stepped_score           distance -> score by a step ladder
    compress_outliers       pull >2 SD values toward 2-3 SD
    compression_parameters  stats for compress_outliers
    normalize_indicators    one indicator -> normalised Series
    build_composite         normalised Series -> Mazziotta-Pareto composite
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Transforms


def soft_threshold(distance, threshold, k=5.0):
    """Distance -> 0-1, half at the threshold. A missing distance (an element not
    measured, such as a block without residents) stays missing."""
    d = pd.to_numeric(distance, errors="coerce").astype("float64")
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(k * (d - threshold) / threshold))


def stepped_score(distance, steps, beyond=0.0):
    """Distance -> the score of the first step it is within, else ``beyond``. A
    missing distance stays missing."""
    d = pd.to_numeric(distance, errors="coerce").astype("float64")
    out = pd.Series(beyond, index=d.index, dtype="float64")
    for limit, value in reversed(steps):
        out[d <= limit] = value
    out[d.isna()] = np.nan
    return out


# ---------------------------------------------------------------------------
# Outlier compression


def compression_parameters(values):
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return {"mean": 0.0, "sd": 0.0, "minimum": 0.0, "maximum": 0.0}
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=0)),
        "minimum": float(v.min()),
        "maximum": float(v.max()),
    }


def compress_outliers(values, mean, sd, minimum, maximum):
    x = pd.to_numeric(values, errors="coerce").astype("float64")
    out = x.copy()
    if not sd > 0:
        return out
    lower, upper = mean - 2 * sd, mean + 2 * sd
    if minimum < lower - sd:
        low = x < lower
        out[low] = lower - sd + sd * (x[low] - minimum) / (lower - minimum)
    if maximum > upper + sd:
        high = x > upper
        out[high] = upper + sd * (x[high] - upper) / (maximum - upper)
    return out


# ---------------------------------------------------------------------------
# Indicator normalisation


def normalize_indicators(
    values,
    polarity="positive",
    method="ampi",
    transform=None,
    reference=None,
    goalposts=None,
    mean=None,
    sd=None,
    compress=False,
):
    """Transform, orient, normalise one indicator. Returns a Series."""
    x = pd.to_numeric(values, errors="coerce").astype("float64")

    if transform:
        if "soft_threshold" in transform:
            x = soft_threshold(
                x,
                float(transform["soft_threshold"]),
                float(transform.get("k", 5.0)),
            )
        elif "steps" in transform:
            steps = [[float(d), float(s)] for d, s in transform["steps"]]
            x = stepped_score(x, steps, float(transform.get("beyond", 0.0)))
        else:
            raise ValueError("transform needs 'soft_threshold' or 'steps'.")

    if compress:
        x = compress_outliers(x, **compression_parameters(x))

    sign = 1.0 if polarity == "positive" else -1.0

    if method == "ampi":
        if reference is None or goalposts is None:
            raise ValueError("ampi needs 'reference' and 'goalposts'.")
        lo, hi = float(goalposts["min"]), float(goalposts["max"])
        if not hi > lo:
            raise ValueError("goalposts must have max > min.")
        return 100 + sign * 60.0 * (x - float(reference)) / (hi - lo)

    if method == "mpi":
        if mean is None or sd is None:
            raise ValueError("mpi needs 'mean' and 'sd'.")
        if not float(sd) > 0:
            raise ValueError("'sd' must be positive.")
        return 100 + sign * 10.0 * (x - float(mean)) / float(sd)

    raise ValueError(f"Unknown method '{method}'.")


# ---------------------------------------------------------------------------
# Composite aggregation (Mazziotta-Pareto)


def build_composite(
    indicators,
    weights=None,
    phenomenon="positive",
    min_valid=None,
):
    """Weighted Mazziotta-Pareto over already-normalised Series.

    Returns DataFrame with columns: index, mean, penalty, n.
    """
    if not indicators:
        raise ValueError("Need at least one indicator.")
    if weights is None:
        weights = [1.0] * len(indicators)
    if len(weights) != len(indicators):
        raise ValueError("weights and indicators must be the same length.")

    frame = pd.concat(indicators, axis=1)
    v = frame.to_numpy(dtype="float64")
    w = np.asarray(weights, dtype="float64")
    valid = ~np.isnan(v)
    n = valid.sum(axis=1)
    raw = np.where(valid, w[None, :], 0.0)
    total = raw.sum(axis=1)
    filled = np.where(valid, v, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = raw / total[:, None]
        mean = (shares * filled).sum(axis=1)
        sd = np.sqrt((shares * (filled - mean[:, None]) ** 2).sum(axis=1))
        penalty = sd * (sd / mean)
    if phenomenon == "positive":
        index = mean - penalty
    elif phenomenon == "negative":
        index = mean + penalty
    else:
        raise ValueError("phenomenon must be 'positive' or 'negative'.")
    required = v.shape[1] if min_valid is None else min_valid
    missing = (n < required) | ~(total > 0)
    index[missing] = np.nan
    mean[missing] = np.nan
    penalty[missing] = np.nan
    return pd.DataFrame(
        {"index": index, "mean": mean, "penalty": penalty, "n": n},
        index=frame.index,
    )
