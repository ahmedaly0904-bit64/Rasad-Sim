import math

import numpy as np
import pytest

from rasad.analyzer import THRESHOLDS, classify, divergence, summarize


def test_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite values"):
        summarize([float("inf"),0.1])
    with pytest.raises(ValueError, match="non-finite values"):
        summarize([float("-inf"),3])
    with pytest.raises(ValueError, match="non-finite values"):
        summarize([float("nan"),1.0])

def test_constant_values_have_zero_spread():
    result = summarize([7.0] * 50)
    assert result["std"] == 0.0
    assert result["cv"] == 0.0
    assert result["mean"] == 7.0
    assert result["min"] == 7.0
    assert result["max"] == 7.0


def test_cv_is_infinite_when_mean_is_zero():
    result = summarize([-1.0, 1.0])
    assert result["mean"] == 0.0
    assert math.isinf(result["cv"])


def test_std_estimates_known_sigma():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=10.0, scale=2.0, size=20000)
    result = summarize(values)
    assert abs(result["std"] - 2.0) < 0.05
    assert abs(result["mean"] - 10.0) < 0.05


def test_percentile_interval_covers_ninety_percent():
    values = list(range(101))
    result = summarize(values)
    assert result["p05"] == pytest.approx(5.0)
    assert result["p95"] == pytest.approx(95.0)


def test_reports_the_number_of_values():
    assert summarize([1.0, 2.0, 3.0])["n"] == 3


def test_requires_at_least_two_values():
    with pytest.raises(ValueError, match="at least 2"):
        summarize([1.0])


def test_classify_boundaries():
    assert classify(0.0) == "low"
    assert classify(0.049) == "low"
    assert classify(THRESHOLDS["low"]) == "moderate"
    assert classify(0.20) == "moderate"
    assert classify(0.201) == "high"
    assert classify(math.inf) == "high"


def test_classify_rejects_nan_instead_of_calling_it_high():
    """Every comparison with nan is False, so it fell through to "high": a broken run turned into a confident verdict. inf stays valid — it is the documented cv of a zero mean with spread."""
    with pytest.raises(ValueError, match="nan"):
        classify(math.nan)


def test_identical_series_never_diverge():
    series = [[1.0, 2.0, 3.0, 4.0]] * 10
    assert divergence(series) == [0.0, 0.0, 0.0, 0.0]


def test_divergence_grows_like_sqrt_time_for_random_walk():
    rng = np.random.default_rng(7)
    steps = 401
    walks = [np.cumsum(rng.normal(0.0, 1.0, steps)).tolist() for _ in range(4000)]

    spread = divergence(walks)
    ratio = spread[400] / spread[100]
    assert abs(ratio - 2.0) < 0.15


def test_divergence_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="same length"):
        divergence([[1.0, 2.0], [1.0, 2.0, 3.0]])


def test_divergence_needs_at_least_two_series():
    with pytest.raises(ValueError, match="at least 2"):
        divergence([[1.0, 2.0]])


def test_divergence_rejects_empty_series():
    with pytest.raises(ValueError, match="at least one time step"):
        divergence([[], []])


def test_divergence_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite values"):
        divergence([[1.0, float("nan")], [1.0, 2.0]])
    with pytest.raises(ValueError, match="non-finite values"):
        divergence([[1.0, float("inf")], [1.0, 2.0]])
    with pytest.raises(ValueError, match="non-finite values"):
        divergence([[1.0, float("-inf")], [1.0, 2.0]])


def test_divergence_and_summarize_reject_non_finite_alike():
    """The same value must not be rejected as a scalar and accepted as a series."""
    with pytest.raises(ValueError, match="non-finite values"):
        summarize([1.0, float("nan")])
    with pytest.raises(ValueError, match="non-finite values"):
        divergence([[1.0, float("nan")], [1.0, 2.0]])


def test_cv_is_zero_when_the_value_never_varies_at_zero():
    """A value constant at zero is fully determined — there is no spread at all.

    Discovered on real data: the Omran total_famines output was zero in every
    run, giving cv = inf and classifying it "high" when it was the most
    stable value in the report.
    """
    result = summarize([0.0] * 10)
    assert result["std"] == 0.0
    assert result["cv"] == 0.0
    assert classify(result["cv"]) == "low"


def test_classify_accepts_custom_thresholds():
    strict = {"low": 0.01, "moderate": 0.05}
    assert classify(0.02) == "low"
    assert classify(0.02, strict) == "moderate"


def test_validate_thresholds_returns_plain_floats():
    from rasad.analyzer import validate_thresholds

    out = validate_thresholds({"low": 0.1, "moderate": 0.3})
    assert out == {"low": 0.1, "moderate": 0.3}
    assert all(type(v) is float for v in out.values())


def test_validate_thresholds_copies_so_the_caller_cannot_mutate_it():
    from rasad.analyzer import validate_thresholds

    original = {"low": 0.1, "moderate": 0.3}
    out = validate_thresholds(original)
    original["low"] = 999.0
    assert out["low"] == 0.1


def test_validate_thresholds_rejects_wrong_keys():
    from rasad.analyzer import validate_thresholds

    with pytest.raises(ValueError, match="keys"):
        validate_thresholds({"low": 0.1})
    with pytest.raises(ValueError, match="keys"):
        validate_thresholds({"low": 0.1, "moderate": 0.3, "extra": 1.0})


def test_validate_thresholds_rejects_non_positive_values():
    from rasad.analyzer import validate_thresholds

    with pytest.raises(ValueError, match="positive"):
        validate_thresholds({"low": 0.0, "moderate": 0.3})
    with pytest.raises(ValueError, match="positive"):
        validate_thresholds({"low": -0.1, "moderate": 0.3})


def test_validate_thresholds_rejects_an_inverted_order():
    from rasad.analyzer import validate_thresholds

    with pytest.raises(ValueError, match="low must be below moderate"):
        validate_thresholds({"low": 0.5, "moderate": 0.2})

def test_validate_thresholds_rejects_equal_thresholds():
    from rasad.analyzer import validate_thresholds

    with pytest.raises(ValueError, match="low must be below moderate"):
        validate_thresholds({"low": 0.2, "moderate": 0.2})


def test_standard_error_is_the_std_over_root_n():
    result = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["se"] == pytest.approx(result["std"] / math.sqrt(5))


def test_a_constant_sample_has_no_uncertainty_about_its_mean():
    result = summarize([7.0] * 50)
    assert result["se"] == 0.0
    assert result["ci_low"] == 7.0
    assert result["ci_high"] == 7.0


def test_the_mean_interval_is_far_tighter_than_the_sample_spread():
    """The two intervals answer different questions: p05-p95 is where a
    single run lands, ci_low-ci_high is where the mean sits."""
    rng = np.random.default_rng(0)
    result = summarize(rng.normal(100.0, 15.0, 500).tolist())
    spread = result["p95"] - result["p05"]
    interval = result["ci_high"] - result["ci_low"]
    assert interval < spread / 5


def test_the_mean_interval_brackets_the_true_mean():
    rng = np.random.default_rng(1)
    result = summarize(rng.normal(100.0, 15.0, 500).tolist())
    assert result["ci_low"] < 100.0 < result["ci_high"]


def test_the_bootstrap_agrees_with_normal_theory_on_symmetric_data():
    """Not a tautology: it is the check that the resampling is wired up
    right. The two must diverge on skewed data, which is why the interval
    is a bootstrap and not mean +/- 1.645 * se."""
    rng = np.random.default_rng(0)
    result = summarize(rng.normal(100.0, 15.0, 500).tolist())
    interval = result["ci_high"] - result["ci_low"]
    assert interval == pytest.approx(2 * 1.645 * result["se"], rel=0.1)


def test_a_sample_larger_than_one_bootstrap_chunk_still_summarizes():
    """The bootstrap resamples in bounded chunks so peak memory does not grow
    with the run count. Crossing a chunk boundary must not change the answer."""
    rng = np.random.default_rng(2)
    values = rng.normal(50.0, 5.0, 5000).tolist()
    result = summarize(values)
    assert result["ci_low"] < 50.0 < result["ci_high"]
    assert summarize(values) == result


def test_summarize_is_a_pure_function_of_its_input():
    """The bootstrap must not make the summary vary between calls."""
    values = [1.0, 5.0, 2.0, 8.0, 3.0, 13.0, 21.0]
    assert summarize(values) == summarize(values)

def test_validate_thresholds_rejects_a_non_numeric_value_as_a_value_error():
    """math.isfinite raises TypeError on a str or None, so a caller guarding measure() with except ValueError — the type both docstrings promise — was crashed instead."""
    from rasad.analyzer import validate_thresholds

    with pytest.raises(ValueError, match="finite positive number"):
        validate_thresholds({"low": "0.05", "moderate": 0.3})
    with pytest.raises(ValueError, match="finite positive number"):
        validate_thresholds({"low": 0.05, "moderate": None})


def test_validate_thresholds_rejects_a_boolean_value():
    """True passed the numeric gate and became 1.0; the adapter rejects booleans as numbers everywhere else."""
    from rasad.analyzer import validate_thresholds

    with pytest.raises(ValueError, match="finite positive number"):
        validate_thresholds({"low": True, "moderate": 0.3})


def test_measure_reports_a_bad_threshold_type_as_a_value_error():
    from reference_models import constant_model

    import rasad

    with pytest.raises(ValueError, match="finite positive number"):
        rasad.measure(constant_model, {}, runs=3, thresholds={"low": "0.05", "moderate": 0.3})
