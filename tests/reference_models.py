"""Reference models whose behaviour is known analytically.

Each one exists so a statistical claim can be checked against a known
answer instead of against whatever the code currently produces.
"""

import numpy as np


def constant_model(params: dict, seed: int) -> dict:
    """Always returns the same number. Expect: standard deviation exactly zero."""
    return {"value": 42.0}


def seed_echo_model(params: dict, seed: int) -> dict:
    """Returns the seed itself. Expect: every run differs, no repeats."""
    return {"value": float(seed)}


def normal_model(params: dict, seed: int) -> dict:
    """Draws from a normal distribution with a known sigma from params."""
    rng = np.random.default_rng(seed)
    return {"value": float(rng.normal(params["mu"], params["sigma"]))}


def flat_series_model(params: dict, seed: int) -> dict:
    """A constant series. Expect: the divergence curve is flat at zero."""
    return {"trace": [1.0, 2.0, 3.0, 4.0, 5.0]}


def random_walk_model(params: dict, seed: int) -> dict:
    """A random walk. Expect: divergence grows as the square root of time."""
    rng = np.random.default_rng(seed)
    steps = params["steps"]
    return {"trace": np.cumsum(rng.normal(0.0, 1.0, steps)).tolist()}


def shape_drift_model(params: dict, seed: int) -> dict:
    """Returns one key as a number on even seeds and a series on odd ones.
    Expect: measure() rejects it rather than summarizing half the runs."""
    return {"v": [1.0, 2.0] if seed % 2 else 3.0}


def tiny_value_model(params: dict, seed: int) -> dict:
    """Draws values around 0.001. Expect: the summary shows them, not 0.00."""
    rng = np.random.default_rng(seed)
    return {"value": float(rng.normal(0.001, 0.0002))}


def empty_series_model(params: dict, seed: int) -> dict:
    """A series with no time steps at all.
    Expect: rejected, not rendered as a zero-length divergence curve."""
    return {"trace": []}


def nan_series_model(params: dict, seed: int) -> dict:
    """Puts a NaN in one run's series only.
    Expect: rejected, exactly as a NaN scalar already is."""
    return {"trace": [1.0, float("nan") if seed == 2 else float(seed), 3.0]}


def mixed_key_model(params: dict, seed: int) -> dict:
    """Names one output with an int and one with a str. Nothing requires
    output names to be strings. Expect: measured like any other model."""
    return {1: float(seed), "a": float(seed * 2)}
