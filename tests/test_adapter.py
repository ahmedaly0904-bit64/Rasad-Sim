import numpy as np
import pytest
from reference_models import constant_model, random_walk_model

from rasad.adapter import split_outputs, validate_model


def test_accepts_a_conforming_model():
    validate_model(constant_model)
    validate_model(random_walk_model)


def test_rejects_a_non_callable():
    with pytest.raises(TypeError, match="callable"):
        validate_model("not a function")


def test_rejects_wrong_parameter_names():
    def bad(config, rng_seed):
        return {}

    with pytest.raises(TypeError, match="params.*seed"):
        validate_model(bad)


def test_rejects_wrong_parameter_count():
    def bad(params):
        return {}

    with pytest.raises(TypeError, match="params.*seed"):
        validate_model(bad)


def test_splits_scalars_from_series():
    scalars, series = split_outputs({"pop": 500.0, "trace": [1.0, 2.0], "wars": 3})
    assert scalars == {"pop": 500.0, "wars": 3.0}
    assert series == {"trace": [1.0, 2.0]}


def test_scalars_are_plain_floats():
    scalars, _ = split_outputs({"wars": 3})
    assert type(scalars["wars"]) is float


def test_rejects_a_boolean_output():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"collapsed": True})


def test_rejects_a_string_output():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"name": "Nation_A"})


def test_accepts_a_numpy_integer_as_a_scalar():
    scalars, series = split_outputs({"wars": np.int64(5)})
    assert scalars == {"wars": 5.0}
    assert type(scalars["wars"]) is float
    assert series == {}


def test_accepts_a_numpy_float_of_any_width():
    scalars, _ = split_outputs({"a": np.float32(1.5), "b": np.float64(2.5)})
    assert scalars == {"a": 1.5, "b": 2.5}
    assert all(type(v) is float for v in scalars.values())


def test_accepts_a_one_dimensional_numpy_array_as_a_series():
    _, series = split_outputs({"trace": np.array([1.0, 2.0, 3.0])})
    assert series == {"trace": [1.0, 2.0, 3.0]}
    assert all(type(v) is float for v in series["trace"])


def test_accepts_a_zero_dimensional_numpy_array_as_a_scalar():
    scalars, series = split_outputs({"total": np.array(7.0)})
    assert scalars == {"total": 7.0}
    assert series == {}


def test_rejects_a_numpy_boolean():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"collapsed": np.bool_(True)})


def test_rejects_a_numpy_boolean_array():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"flags": np.array([True, False])})


def test_rejects_a_multidimensional_array():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"grid": np.array([[1.0, 2.0], [3.0, 4.0]])})


def test_rejects_an_object_dtype_array():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"stuff": np.array(["a", "b"], dtype=object)})


def test_still_rejects_a_python_boolean():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"collapsed": True})


def test_rejects_a_list_of_python_booleans():
    """A list of booleans is swallowed as ones and zeros, producing meaningless statistics — the same trap as a single boolean value."""
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"alive": [True, False, True]})


def test_rejects_a_list_of_strings_with_a_clear_message():
    """Without the element check, float() would raise a confusing ValueError instead of a clear message."""
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"names": ["a", "b"]})


def test_rejects_a_list_of_nested_lists():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"grid": [[1.0, 2.0], [3.0, 4.0]]})


def test_rejects_a_numpy_complex_scalar():
    """A complex value silently loses its imaginary part to float(), so every statistic below would describe the real part alone."""
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"amplitude": np.complex128(1 + 2j)})


def test_rejects_a_numpy_complex_array():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"wave": np.array([1 + 2j, 3 + 4j])})


def test_rejects_a_list_of_numpy_complex_values():
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"wave": [np.complex128(1 + 2j), np.complex128(3 + 4j)]})


def test_rejects_a_timedelta_output_by_naming_the_key():
    """timedelta64 counts as a numpy integer, so it passed the numeric gate and failed later in float() with a message that never named the output."""
    with pytest.raises(TypeError, match="unsupported output"):
        split_outputs({"elapsed": np.timedelta64(5, "D")})
