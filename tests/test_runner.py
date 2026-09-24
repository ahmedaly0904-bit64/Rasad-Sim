import pytest
from reference_models import constant_model, seed_echo_model

from rasad.runner import run_all
from rasad.sampler import make_plan


def test_collects_one_output_dict_per_run():
    plan = make_plan({}, runs=3)
    assert run_all(constant_model, plan) == [{"value": 42.0}] * 3


def test_passes_each_seed_to_the_model():
    plan = make_plan({}, runs=3, base_seed=50)
    results = run_all(seed_echo_model, plan)
    assert [r["value"] for r in results] == [50.0, 51.0, 52.0]


def test_preserves_plan_order():
    plan = make_plan({}, runs=5, base_seed=10)
    results = run_all(seed_echo_model, plan)
    assert [r["value"] for r in results] == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_validates_the_model_before_running():
    with pytest.raises(TypeError, match="callable"):
        run_all("not a function", make_plan({}, runs=2))


def test_rejects_a_model_that_does_not_return_a_dict():
    def bad(params, seed):
        return 42

    with pytest.raises(TypeError, match="must return a dict"):
        run_all(bad, make_plan({}, runs=2))


def test_rejects_inconsistent_output_keys():
    def unstable(params, seed):
        return {"a": 1.0} if seed % 2 == 0 else {"b": 1.0}

    with pytest.raises(ValueError, match="same output keys"):
        run_all(unstable, make_plan({}, runs=2))


def test_inconsistent_keys_of_mixed_types_are_reported_not_crashed_on():
    def unstable(params, seed):
        return {1: 1.0, "a": 1.0} if seed % 2 == 0 else {"b": 1.0}

    with pytest.raises(ValueError, match="same output keys"):
        run_all(unstable, make_plan({}, runs=2))


def test_a_model_failure_carries_the_seed_that_caused_it():
    def crash_on_seed_3(params, seed):
        if seed == 3:
            raise RuntimeError("boom")
        return {"value": float(seed)}

    with pytest.raises(RuntimeError) as excinfo:
        run_all(crash_on_seed_3, make_plan({}, runs=10))

    assert "boom" in str(excinfo.value)
    assert any("seed 3" in note for note in excinfo.value.__notes__)
