from __future__ import annotations

import pytest

from numa.vector import Vector


def test_comparing_with_an_another_type_is_false():
    assert Vector([1]) != 1


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
    assert Vector([1, 2, 3]).dot(Vector([5, 7, 11])) == 5 + 14 + 33


def test_dot_product_of_two_vectors_of_different_shape_raise_a_value_error():
    with pytest.raises(ValueError):
        assert Vector([1]).dot(Vector([1, 1]))


def test_multiplying_by_an_integer():
    assert Vector([1, 2, 3]) * 2 == Vector([2, 4, 6])


def test_multiplying_by_a_float():
    assert Vector([1, 2, 3]) * 2.0 == Vector([2.0, 4.0, 6.0])


def test_multiplying_by_an_invalid_type_raise_a_value_error():
    with pytest.raises(ValueError):
        assert Vector([1]) * True


def test_linear_space_factory():
    assert Vector.linspace(0, 1, 3) == Vector([0, 0.5, 1])


def test_creating_a_vector_of_zeros():
    assert Vector.zeros(3) == Vector([0, 0, 0])


def test_creating_a_vector_of_zeros_except_a_positive_size():
    with pytest.raises(ValueError):
        Vector.zeros(-1)


def test_iterate_through_vectors():
    assert list(Vector([1, 2, 3])) == [1, 2, 3]


def test_accessing_element_by_index():
    assert Vector([1, 2, 3])[1] == 2


def test_powering_vectors():
    assert Vector([1, 2, 3]) ** 2 == Vector([1, 4, 9])


def test_powering_vectors_by_0_give_the_unit_vector():
    assert Vector([1, 2, 3]) ** 0 == Vector([1, 1, 1])


def test_multiplying_vectors():
    assert Vector([1, 2, 3]).multiply(Vector([1, 2, 3])) == Vector([1, 4, 9])


def test_multiplying_vectors_of_different_size_raise_a_value_error():
    with pytest.raises(ValueError):
        Vector([1, 2]).multiply(Vector([1]))
