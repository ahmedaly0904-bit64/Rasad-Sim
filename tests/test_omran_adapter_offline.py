"""Adapter behaviour that does not need the real Omran.

A stand-in ``nation``/``world`` pair is written to a temporary directory so
these checks run everywhere, including CI where Omran is absent.
"""

import random
import sys

import pytest

from rasad.adapters.omran import make_omran_run

_NATION = '''
import random


class Nation:
    def __init__(self, name, population, food, growth_rate):
        self.name = name
        self.population = population
        self.is_alive = True
        self.war_count = 0
        self.famine_count = 0
        random.random()
'''

_WORLD = '''
class WorldModel:
    def __init__(self, nations):
        self.nations = nations

    def step(self):
        pass
'''


@pytest.fixture
def stand_in_omran(tmp_path, monkeypatch):
    (tmp_path / "nation.py").write_text(_NATION)
    (tmp_path / "world.py").write_text(_WORLD)
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {name: sys.modules.pop(name) for name in ("nation", "world") if name in sys.modules}
    yield str(tmp_path)
    for name in ("nation", "world"):
        sys.modules.pop(name, None)
    sys.modules.update(saved)


@pytest.mark.parametrize(
    "nations",
    [
        [{"name": "X"}],
        [{"name": "X", "population": 1, "food": 1, "growth_rate": 0.1, "extra": 1}],
        ["not a dict"],
        "abc",
    ],
)
def test_a_malformed_nation_spec_does_not_touch_the_global_rng(stand_in_omran, nations):
    run = make_omran_run(stand_in_omran, years=3)
    random.seed(12345)
    before = random.random()
    random.seed(12345)
    with pytest.raises(TypeError):
        run({"nations": nations}, 999)
    assert random.random() == before


def test_a_valid_run_is_still_seeded(stand_in_omran):
    run = make_omran_run(stand_in_omran, years=3)
    run({}, 7)
    after_first = random.random()
    run({}, 7)
    assert random.random() == after_first
