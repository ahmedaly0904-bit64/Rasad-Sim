"""Arithmetic on sequences of numbers, nothing more.

This module knows nothing about simulations or how the numbers it is
given were produced. It only describes the spread and location of a
sample: count, mean, sample standard deviation, coefficient of
variation, extrema, and percentile interval — and, separately, how
well that sample pins its own mean: the standard error and a
bootstrap confidence interval.
"""

import math
from collections.abc import Sequence

import numpy as np

# Default classification cutoffs. These are a convention chosen by the
# project's authors, not a derived or theoretical result: a caller may
# pass their own thresholds to ``classify`` and ``rasad.measure``, in
# which case these defaults are ignored.
THRESHOLDS: dict[str, float] = {"low": 0.05, "moderate": 0.20}

# Fixed bootstrap settings so the confidence interval is a pure function of
# the input: a seeded generator always yields the same resample draws, so the
# same values in give byte-identical ``ci_low`` / ``ci_high`` out, in every
# process. The seed is deliberately never drawn from the clock or from global
# numpy state.
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_SEED = 0
# Resample in chunks of at most this many elements. A single (resamples, n)
# draw would allocate memory linear in the run count — 1.5 GB at n=10,000,
# a run count this library's own docs contemplate. Chunking bounds the peak
# regardless of n, and numpy's generator consumes its bit stream value by
# value, so the chunked draws are the identical sequence: same numbers out.
_BOOTSTRAP_CHUNK_ELEMENTS = 4_000_000


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
        ``ci_low``/``ci_high`` bound a 90% bootstrap percentile confidence
        interval on the mean. ``std``, ``cv``, ``p05`` and ``p95``, by
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
    if std == 0.0:
        ci_low = ci_high = mean
    else:
        rng = np.random.default_rng(_BOOTSTRAP_SEED)
        rows = max(1, _BOOTSTRAP_CHUNK_ELEMENTS // n)
        means = np.empty(_BOOTSTRAP_RESAMPLES)
        for start in range(0, _BOOTSTRAP_RESAMPLES, rows):
            size = min(rows, _BOOTSTRAP_RESAMPLES - start)
            means[start : start + size] = arr[
                rng.integers(0, n, size=(size, n))
            ].mean(axis=1)
        ci_low = float(np.percentile(means, 5))
        ci_high = float(np.percentile(means, 95))

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
        values must be finite ``int`` or ``float`` values greater than 0
        — never a ``bool`` — and ``low`` must be strictly below
        ``moderate``.

    Returns
    -------
    dict[str, float]
        A plain-float copy of ``thresholds``, so mutating the caller's
        dict afterwards cannot change the classification.

    Raises
    ------
    ValueError
        When the keys, the type, the positivity/finiteness, or the
        ordering of the values do not satisfy the rules above. A
        non-numeric value is reported as a ``ValueError`` too, so one
        ``except ValueError`` catches every way the cutoffs can be wrong.
    """
    keys = set(thresholds)
    if keys != {"low", "moderate"}:
        raise ValueError(
            f"thresholds must have exactly the keys 'low' and 'moderate', got {sorted(keys)}"
        )
    low = thresholds["low"]
    moderate = thresholds["moderate"]
    for name, value in (("low", low), ("moderate", moderate)):
        # The type is checked here rather than left to math.isfinite, which
        # raises TypeError on a str or None — an exception neither this
        # function nor rasad.measure documents, so a caller guarding the call
        # with ``except ValueError`` was crashed by bad configuration instead
        # of told about it. A bool is excluded for the same reason the adapter
        # excludes it from numbers: True is not a cutoff.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} threshold must be a finite positive number, got {value!r}")
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
