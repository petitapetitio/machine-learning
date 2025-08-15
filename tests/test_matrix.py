"""
Opérations
- addition
- multiplication par un scalaire
- inversion

Factories
- identité
- créer une matrice à partir d'une liste de vecteurs

- Ajouter une colonne
- Créer une matrice à partir de 2 vecteurs
"""

from __future__ import annotations

import pytest

from numa.matrix import Matrix
from numa.vector import Vector


def test_comparison():
    assert Matrix([Vector([1])]) == Matrix([Vector([1])])


def test_comparison_with_another_type_is_false():
    assert not Vector([1]) == 1


def test_create_matrix_from_vectors():
    assert Matrix.with_columns(Vector([1, 2, 3]), Vector([4, 5, 6])) == Matrix(
        [
            Vector([1, 4]),
            Vector([2, 5]),
            Vector([3, 6]),
        ]
    )


def test_create_matrix_from_vectors_of_different_size_raise_a_value_error():
    with pytest.raises(ValueError):
        Matrix.with_columns(Vector([1]), Vector([4, 5]))


def test_column_are_indexed_from_zero():
    assert Matrix.with_columns(Vector([1, 2, 3])).column(0) == Vector([1, 2, 3])


def test_accessing_column_by_invalid_index_raise_index_error():
    with pytest.raises(IndexError):
        Matrix.with_columns(Vector([1, 2, 3])).column(1)


def test_accessing_columns():
    assert list(Matrix.with_columns(Vector([1, 2]), Vector([4, 5])).columns()) == [
        Vector([1, 2]),
        Vector([4, 5]),
    ]


def test_transposing():
    column1 = Vector([1, 2])
    column2 = Vector([4, 5])
    assert Matrix.with_rows(rows=[column1, column2]).transpose() == Matrix.with_columns(column1, column2)
