"""Arithmetic on sequences of numbers, nothing more.

This module knows nothing about simulations or how the numbers it is
given were produced. It only describes the spread and location of a
sample: count, mean, sample standard deviation, coefficient of
variation, extrema, and percentile interval — and, separately, how
well that sample pins its own mean: the standard error and a
Student-t confidence interval.
"""

import math
from collections.abc import Sequence

import numpy as np

# Default classification cutoffs. These are a convention chosen by the
# project's authors, not a derived or theoretical result: a caller may
# pass their own thresholds to ``classify`` and ``rasad.measure``, in
# which case these defaults are ignored.
THRESHOLDS: dict[str, float] = {"low": 0.05, "moderate": 0.20}

# The confidence level of the interval on the mean, as the one-sided upper
# probability: 0.95 leaves 5% in each tail, a 90% two-sided interval.
_CI_UPPER = 0.95


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _t_two_sided_tail(t: float, df: int) -> float:
    """P(|T| > t) for Student's t with ``df`` degrees of freedom."""
    a, b, x = df / 2.0, 0.5, df / (df + t * t)
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def t_quantile(p: float, df: int) -> float:
    """The ``p`` quantile of Student's t distribution, for ``p`` above 0.5.

    Computed by bisection on the exact distribution function, so the
    library needs nothing beyond the standard library and numpy. Agrees
    with ``scipy.stats.t.ppf`` to better than one part in 10^8.

    Raises
    ------
    ValueError
        When ``p`` is not strictly between 0.5 and 1, or ``df`` is below 1.
    """
    if not 0.5 < p < 1.0:
        raise ValueError(f"p must be strictly between 0.5 and 1, got {p!r}")
    if df < 1:
        raise ValueError(f"df must be at least 1, got {df!r}")
    tail = 2.0 * (1.0 - p)
    lo, hi = 0.0, 1.0
    while _t_two_sided_tail(hi, df) > tail:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_two_sided_tail(mid, df) > tail:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    return (lo + hi) / 2.0


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize a sample of numbers.

    Parameters
    ----------
    values
        A sequence of numeric values, e.g. repeated simulation runs.

    Returns
    -------
    dict
        Keys ``n``, ``mean``, ``std``, ``cv``, ``min``, ``max``, ``p05``,
        ``p95``, ``se``, ``ci_low``, ``ci_high``. ``std`` is the sample
        standard deviation (``ddof=1``); ``cv`` is ``std / abs(mean)``; it
        is zero whenever ``std`` is zero, and infinite when the mean is zero
        but the values still vary.

        ``se``, ``ci_low`` and ``ci_high`` describe the uncertainty of the
        mean: ``se`` is the standard error, ``std / sqrt(n)``, and
        ``ci_low``/``ci_high`` bound a 90% Student-t confidence interval on
        the mean, ``mean ± t(0.95, n - 1) * se``. The t multiplier is what
        keeps the stated 90% honest at the small run counts this library
        allows: with two runs it is 6.31, not the 1.645 of large samples,
        so the interval is as wide as two runs deserve. ``std``, ``cv``, ``p05`` and ``p95``, by
        contrast, describe the spread of the sample: where a single run
        lands. The two answer different questions — how well the runs pin
        the average versus how far apart the runs lie — and are not
        interchangeable.

    Raises
    ------
    ValueError
        When fewer than two values are provided, or when any value is
        not finite. :func:`divergence` rejects a non-finite series with
        the same message, so one phrase matches either path.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        raise ValueError("need at least 2 values to summarize")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values")

    n = int(arr.size)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # The order of these checks is deliberate: zero spread means a fully
    # determined value whatever the mean. Without this, an output constantly
    # at zero would be classified "high" when it is the most stable one
    # there is.
    if std == 0.0:
        cv = 0.0
    elif mean == 0.0:
        cv = math.inf
    else:
        cv = std / abs(mean)
    p05 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    se = std / math.sqrt(n)
    half_width = t_quantile(_CI_UPPER, n - 1) * se
    ci_low = mean - half_width
    ci_high = mean + half_width

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": p05,
        "p95": p95,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def validate_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Check caller-supplied classification cutoffs and copy them.

    The thresholds are a convention, not a measured property of the data,
    so whatever the caller passes is taken as-is once it passes these
    checks.

    Parameters
    ----------
    thresholds
        A dict with exactly the keys ``low`` and ``moderate``. Both
        values must be finite numbers greater than 0, and ``low``
        must be strictly below ``moderate``.

    Returns
    -------
    dict[str, float]
        A plain-float copy of ``thresholds``, so mutating the caller's
        dict afterwards cannot change the classification.

    Raises
    ------
    ValueError
        When the keys, the positivity/finiteness, or the ordering of the
        values do not satisfy the rules above.
    """
    keys = set(thresholds)
    if keys != {"low", "moderate"}:
        raise ValueError(
            f"thresholds must have exactly the keys 'low' and 'moderate', got {sorted(keys)}"
        )
    low = thresholds["low"]
    moderate = thresholds["moderate"]
    for name, value in (("low", low), ("moderate", moderate)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} threshold must be a finite positive number, got {value!r}")
    if not low < moderate:
        raise ValueError("low must be below moderate")
    return {"low": float(low), "moderate": float(moderate)}


def classify(cv: float, thresholds: dict[str, float] | None = None) -> str:
    """Label a coefficient of variation by its variability cutoff.

    Parameters
    ----------
    cv
        The coefficient of variation of a sample.
    thresholds
        The classification cutoffs. ``None`` means use the module-level
        :data:`THRESHOLDS` default; any other dict is validated with
        :func:`validate_thresholds` first.

    Returns
    -------
    str
        ``"low"`` below the low threshold, ``"moderate"`` up to and
        including the moderate threshold, ``"high"`` beyond it.
    """
    # kept as if/else rather than a ternary: the two branches do different
    # things — one picks a default, the other validates untrusted input.
    if thresholds is None:  # noqa: SIM108
        thresholds = THRESHOLDS
    else:
        thresholds = validate_thresholds(thresholds)
    if cv < thresholds["low"]:
        return "low"
    if cv <= thresholds["moderate"]:
        return "moderate"
    return "high"


def divergence(series: Sequence[Sequence[float]]) -> list[float]:
    """Measure how far apart a set of time series has drifted by time step.

    At each time step the runs are treated as a sample and the sample
    standard deviation is taken across runs; a flat zero curve means the
    runs agree, a growing curve means they are drifting apart.

    Parameters
    ----------
    series
        A sequence of equal-length time series, one per simulation run.

    Returns
    -------
    list[float]
        One sample standard deviation per time step, across runs.

    Raises
    ------
    ValueError
        When fewer than two series are provided, when the series do
        not all have the same length, when a series has no time
        steps, or when a series contains a non-finite value.
    """
    rows = [np.asarray(row, dtype=float) for row in series]
    if len(rows) < 2:
        raise ValueError("need at least 2 series to measure divergence")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("all series must have the same length")
    if len(rows[0]) == 0:
        raise ValueError("each series must have at least one time step")

    stack = np.stack(rows)
    if not np.isfinite(stack).all():
        raise ValueError("non-finite values")
    return [float(std) for std in stack.std(axis=0, ddof=1)]
