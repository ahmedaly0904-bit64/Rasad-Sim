"""Wrap Omran from the outside so Rasad can measure it.

This file lives in the Rasad repository. It imports Omran's ``nation`` and
``world`` modules by putting Omran's ``src`` directory on ``sys.path`` at
call time, and drives Omran purely through its public interface. Nothing
inside the Omran project is created, modified, or deleted.
"""

import contextlib
import io
import random
import sys
from collections.abc import Callable

_DEFAULT_NATIONS = (
    {"name": "Nation_A", "population": 500, "food": 2000, "growth_rate": 0.03},
    {"name": "Nation_B", "population": 80, "food": 2000, "growth_rate": 0.035},
    {"name": "Nation_C", "population": 200, "food": 2000, "growth_rate": 0.032},
)


def make_omran_run(omran_src: str, years: int) -> Callable[[dict, int], dict]:
    """Build a Rasad-compatible model function that runs Omran.

    Parameters
    ----------
    omran_src
        Absolute path to Omran's ``src`` directory. It is inserted at the
        front of ``sys.path`` if not already present, so that ``nation``
        and ``world`` can be imported from outside the Omran project.
    years
        How many simulated years each run performs.

    Returns
    -------
    Callable[[dict, int], dict]
        A closure ``run(params, seed)`` that seeds Python's global RNG,
        builds a fresh set of nations, steps Omran's ``WorldModel`` for
        ``years`` years with stdout suppressed, and returns Omran's output
        as scalars plus a population trace. The global RNG state is saved
        before seeding and restored afterwards, so a run is reproducible
        from its seed without leaving the caller's randomness reseeded. ``params`` accepts a single
        key, ``nations``: a list of dicts, each holding the keyword
        arguments for Omran's ``Nation(name, population, food,
        growth_rate)``. When the key is absent, the three defaults in
        ``_DEFAULT_NATIONS`` are used. A fresh ``Nation`` is built on
        every run; instances are never reused between runs.

    Raises
    ------
    ValueError
        When the returned ``run`` is called with a ``params`` dict that
        contains any key other than ``nations``. The offending keys are
        named in the message so a setting can never be silently ignored.
    """
    if omran_src not in sys.path:
        sys.path.insert(0, omran_src)

    from nation import Nation
    from world import WorldModel

    def run(params: dict, seed: int) -> dict:
        # Validate before seeding: a rejected call must leave the caller's
        # global RNG exactly as it found it.
        unknown = sorted(set(params) - {"nations"})
        if unknown:
            raise ValueError(
                f"Unknown params keys: {', '.join(unknown)}. "
                "The only supported key is 'nations'."
            )

        # Omran draws from the random module's global generator, so seeding
        # it is the only way to control a run. The caller's state is saved
        # and put back afterwards: without this, measuring a model would
        # leave the whole process reseeded at base_seed + runs - 1, quietly
        # taking over the randomness of any code that runs later.
        rng_state = random.getstate()
        try:
            random.seed(seed)

            nations_spec = params.get("nations", _DEFAULT_NATIONS)
            nations = [Nation(**spec) for spec in nations_spec]

            population_trace = []
            with contextlib.redirect_stdout(io.StringIO()):
                world = WorldModel(nations)
                for _ in range(years):
                    world.step()
                    population_trace.append(
                        sum(n.population for n in world.nations if n.is_alive)
                    )
        finally:
            random.setstate(rng_state)

        return {
            "final_total_population": float(
                sum(n.population for n in world.nations if n.is_alive)
            ),
            "survivors": float(sum(1 for n in world.nations if n.is_alive)),
            "total_wars": float(sum(n.war_count for n in world.nations)),
            "total_famines": float(sum(n.famine_count for n in world.nations)),
            "population_trace": [float(value) for value in population_trace],
        }

    return run
