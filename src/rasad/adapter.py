"""Check that a user model fits the required interface and sort its output.

A model is any plain function matching the required simulation
interface::

    def run(params: dict, seed: int) -> dict: ...

This module validates that a candidate function actually has those two
parameters, and splits a returned result dict into scalar results (a
single final number) and time series (lists of numbers), rejecting
anything a simulation should never produce.

Numpy scalars of real numeric dtype and one-dimensional numpy arrays
of real numeric dtype are accepted and converted to plain Python values.
"Real numeric" means integer or floating point only: booleans are never
accepted, whether Python ``bool`` or ``numpy.bool_``, and neither are
complex numbers, whose imaginary part ``float()`` would discard in
silence.
"""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np


def validate_model(fn: Callable[..., dict]) -> None:
    """Check that a function matches the required model interface.

    Parameters
    ----------
    fn
        The candidate model, expected to accept exactly ``params`` and
        ``seed`` as its first two parameters.

    Raises
    ------
    TypeError
        When ``fn`` is not callable, or when its first two parameter
        names are not exactly ``("params", "seed")``.
    """
    if not callable(fn):
        raise TypeError(f"model must be callable, got {type(fn).__name__}")

    names = [p.name for p in inspect.signature(fn).parameters.values()][:2]
    if names != ["params", "seed"]:
        raise TypeError(
            f"model must accept (params, seed) as its first two arguments, got ({', '.join(names)})"
        )


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    """True for an integer or floating-point dtype, and nothing else.

    ``np.issubdtype(dtype, np.number)`` is too wide: it also admits complex
    dtypes, which ``float()`` narrows to the real part with only a warning,
    and ``timedelta64``, which numpy counts as a signed integer. Testing the
    kind directly admits exactly ``i``nteger, ``u``nsigned and ``f``loat.
    """
    return dtype.kind in "iuf"


def _is_number(value: Any) -> bool:
    """True for a real number — never for a boolean, of either flavour.

    ``isinstance(True, int)`` is True in Python and ``np.bool_`` is a numpy
    scalar, so both must be excluded before any numeric test.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, np.generic) and _is_real_numeric_dtype(value.dtype)


def split_outputs(out: dict[str, Any]) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Split a model result into scalar results and time series.

    Parameters
    ----------
    out
        The dict returned by a model run. Each value must be a number
        (a scalar result) or a list/tuple of numbers (a time series).
        Numpy scalars of real numeric dtype and zero-dimensional numpy
        arrays of real numeric dtype count as numbers; one-dimensional
        numpy arrays of real numeric dtype count as time series.

    Returns
    -------
    tuple[dict[str, float], dict[str, list[float]]]
        ``(scalars, series)``: scalars holds each single-number output
        as a plain ``float``, series holds each sequence output as a
        ``list[float]``.

    Raises
    ------
    TypeError
        When a value is a ``bool`` or ``numpy.bool_``, a ``str``,
        ``bytes``, a complex number, or any other type that is neither a
        single real number nor a sequence of real numbers, including numpy
        arrays with two or more dimensions and arrays of boolean, complex
        or object dtype.
    """
    scalars: dict[str, float] = {}
    series: dict[str, list[float]] = {}

    for key, value in out.items():
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"unsupported output {key!r}: boolean values are not allowed")
        if isinstance(value, np.ndarray):
            if value.ndim == 0 and _is_real_numeric_dtype(value.dtype):
                scalars[key] = float(value)
            elif value.ndim == 1 and _is_real_numeric_dtype(value.dtype):
                series[key] = [float(item) for item in value]
            elif np.issubdtype(value.dtype, np.bool_):
                raise TypeError(
                    f"unsupported output {key!r}: boolean arrays are not allowed"
                )
            else:
                raise TypeError(
                    f"unsupported output {key!r}: numpy array with dtype "
                    f"{value.dtype} and {value.ndim} dimension(s) is not supported"
                )
        elif isinstance(value, (int, float)) or (
            isinstance(value, np.generic) and _is_real_numeric_dtype(value.dtype)
        ):
            scalars[key] = float(value)
        elif isinstance(value, (list, tuple)):
            # Every element is checked: a list of booleans or strings would
            # otherwise pass silently and produce statistics over numbers
            # that mean nothing.
            bad = next(
                ((i, v) for i, v in enumerate(value) if not _is_number(v)), None
            )
            if bad is not None:
                index, item = bad
                raise TypeError(
                    f"unsupported output {key!r}: element {index} is "
                    f"{type(item).__name__}, not a number"
                )
            series[key] = [float(item) for item in value]
        else:
            raise TypeError(
                f"unsupported output {key!r}: must be a number or a sequence of numbers"
            )

    return scalars, series
