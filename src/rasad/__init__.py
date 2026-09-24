"""Rasad — measuring the variability of simulation results."""

from collections.abc import Callable
from typing import Any

from rasad.adapter import split_outputs
from rasad.analyzer import (
    THRESHOLDS,
    classify,
    divergence,
    summarize,
    validate_thresholds,
)
from rasad.report import Report
from rasad.runner import run_all
from rasad.sampler import make_plan

__all__ = ["measure", "Report"]


def measure(
    fn: Callable[..., dict],
    params: dict[str, Any],
    runs: int = 100,
    base_seed: int = 0,
    thresholds: dict[str, float] | None = None,
) -> Report:
    """Measure the variability of a simulation model's output.

    Parameters
    ----------
    fn
        A model matching the simulation interface
        ``def run(params: dict, seed: int) -> dict``.
    params
        The simulation parameters, fixed across every run.
    runs
        How many runs to make. Must be at least 2.
    base_seed
        The seed of the first run; run ``i`` gets ``base_seed + i``.
    thresholds
        The classification cutoffs. ``None`` means use the default
        :data:`rasad.analyzer.THRESHOLDS`. These values are a convention
        the caller owns — they describe what counts as "low" /
        "moderate" / "high" variability for this measurement and are not
        derived from the data. A supplied dict is validated before any run.

    Returns
    -------
    Report
        Per scalar output its summary plus a variability label, per series
        output its divergence curve, the run count, and the thresholds used.

    Raises
    ------
    ValueError
        When ``runs`` is below 2, when ``thresholds`` fails
        :func:`rasad.analyzer.validate_thresholds`, or when an output
        changes shape between runs — the same key returned as a number
        in some runs and as a sequence in the others. Also propagates the
        ``ValueError`` raised by :func:`rasad.analyzer.divergence` for a
        series with no time steps or a non-finite value in it.
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    thresholds = validate_thresholds(thresholds)

    plan = make_plan(params, runs, base_seed)
    results = run_all(fn, plan)

    scalar_values: dict[str, list[float]] = {}
    series_curves: dict[str, list[list[float]]] = {}
    for out in results:
        scalars, series = split_outputs(out)
        for name, value in scalars.items():
            scalar_values.setdefault(name, []).append(value)
        for name, curve in series.items():
            series_curves.setdefault(name, []).append(curve)

    for name in sorted(scalar_values.keys() | series_curves.keys(), key=repr):
        n_scalars = len(scalar_values.get(name, ()))
        n_series = len(series_curves.get(name, ()))
        if n_scalars != runs and n_series != runs:
            raise ValueError(
                f"output {name!r} changed shape between runs: it was a single "
                f"number in {n_scalars} of {runs} runs and a sequence in "
                f"{n_series} of them; every output must keep the same shape "
                f"in every run"
            )

    scalars: dict[str, dict[str, Any]] = {}
    for name, values in scalar_values.items():
        stats = summarize(values)
        stats["variability"] = classify(stats["cv"], thresholds)
        scalars[name] = stats

    series = {name: divergence(curves) for name, curves in series_curves.items()}

    return Report(scalars=scalars, series=series, runs=runs, thresholds=thresholds)
