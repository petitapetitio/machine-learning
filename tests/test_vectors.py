from __future__ import annotations

import pytest

from numa.vector import Vector


def test_comparing_with_an_incompatible_type_throw_a_value_error():
    with pytest.raises(ValueError):
        _ = Vector([1]) == 1


def test_comparing_distinct_vectors_return_false():
    assert not Vector([1]) == Vector([2])


def test_comparing_to_equals_vectors_return_true():
    assert Vector([1]) == Vector([1])


def test_adding_two_vectors():
    assert (Vector([1, 2, 3]) + Vector([1, 3, 5])) == Vector([2, 5, 8])


def test_adding_two_vectors_of_different_shape_raise_a_value_error():
    with pytest.raises(ValueError):
        assert Vector([1]) + Vector([1, 2])


def test_adding_a_number_to_a_vector_throw_an_exception():
    with pytest.raises(ValueError):
        Vector([1, 2, 3]) + 4


def test_subtracting_two_vectors():
    assert Vector([2, 5, 8]) - Vector([1, 2, 3]) == Vector([1, 3, 5])


def test_subtracting_a_number_to_a_vector_throw_an_exception():
    with pytest.raises(ValueError):
        Vector([1, 2, 3]) - 4


def test_subtracting_two_vectors_of_different_shape_raise_a_value_error():
    with pytest.raises(ValueError):
        assert Vector([1]) - Vector([1, 2])


def test_dot_product_of_two_vectors():
    assert Vector([1, 2, 3]).dot(Vector([5, 7, 11])) == Vector([5, 14, 33])


def test_dot_product_of_two_vectors_of_different_shape_raise_a_value_error():
    with pytest.raises(ValueError):
        assert Vector([1]).dot(Vector([1, 1]))


def test_multiplying_by_an_integer():
    assert Vector([1, 2, 3]) * 2 == Vector([2, 4, 6])


def test_multiplying_by_a_float():
    assert Vector([1, 2, 3]) * 2.0 == Vector([2., 4., 6.])


def test_multiplying_by_an_invalid_type_raise_a_value_error():
    with pytest.raises(ValueError):
        assert Vector([1]) * True


def test_linear_space_factory():
    assert Vector.linspace(0, 1, 3) == Vector([0, .5, 1])


def test_iterate_through_vectors():
    assert list(Vector([1, 2, 3])) == [1, 2, 3]
