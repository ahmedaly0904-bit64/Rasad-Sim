"""Execute a plan of simulation runs and collect the raw outputs.

The runner turns a plan from :mod:`rasad.sampler` into results: it
calls the model once per plan entry, in plan order, and returns the
output dicts in that same order. Execution is deliberately sequential;
parallelism is explicitly out of scope for this version.

The model is validated up front so a bad model fails immediately
instead of after wasted work, and every run must return the same set of
output keys or the batch is rejected — a report built from shifting keys
could not be summarized.
"""

from collections.abc import Callable

from rasad.adapter import validate_model


def run_all(fn: Callable[..., dict], plan: list[tuple[dict, int]]) -> list[dict]:
    """Run a model once per plan entry and return the outputs in order.

    Parameters
    ----------
    fn
        A model matching the simulation interface
        ``def run(params: dict, seed: int) -> dict``.
    plan
        The list of ``(params, seed)`` pairs produced by
        :func:`rasad.sampler.make_plan`.

    Returns
    -------
    list[dict]
        One output dict per plan entry, in plan order.

    Raises
    ------
    TypeError
        When ``fn`` is not a valid model, or when a run returns
        something other than a dict.
    ValueError
        When runs do not all return the same set of output keys.
    Exception
        Any exception raised by the model propagates unchanged, with a
        note attached naming the seed of the run that raised it.
    """
    validate_model(fn)

    results: list[dict] = []
    for params, seed in plan:
        try:
            out = fn(params, seed)
        except Exception as exc:
            exc.add_note(f"raised by the model during the run with seed {seed}")
            raise
        if not isinstance(out, dict):
            raise TypeError(
                f"model must return a dict, got {type(out).__name__} from seed {seed}"
            )
        if results and out.keys() != results[0].keys():
            raise ValueError(
                "every run must return the same output keys, "
                f"got {sorted(out, key=repr)} after {sorted(results[0], key=repr)}"
            )
        results.append(out)

    return results
