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
from numbers import Number

import pytest

from numa.vector import Vector


class Matrix:
    def __init__(self, rows: list[Vector]):
        self._rows = rows

    @classmethod
    def create(cls, rows: list[Vector]):
        if not _are_all_of_same_size(rows):
            raise ValueError
        return Matrix(rows)

    @classmethod
    def _c(cls, *columns: Vector):
        cols = list(columns)
        if len(cols) == 0:
            return Matrix([])

        if not _are_all_of_same_size(cols):
            raise ValueError

        rows = []
        for i in range(cols[0].size()):
            rows.append(Vector([col[i] for col in cols]))
        return Matrix(rows)

    def __eq__(self, other):
        if isinstance(other, Matrix):
            for r1, r2 in zip(self._rows, other._rows):
                if r1 != r2:
                    return False
            return True
        return False

    def __repr__(self) -> str:
        s = "Matrix([\n"
        for row in self._rows:
            s += "\t" + repr(row) + ",\n"
        s += "])"
        return s


def _are_all_of_same_size(vectors: list[Vector]):
    size = vectors[0].size()
    for col in vectors:
        if col.size() != size:
            return False
    return True


def test_comparison():
    assert Matrix([Vector([1])]) == Matrix([Vector([1])])


def test_comparison_with_another_type_is_false():
    assert not Vector([1]) == 1


def test_create_matrix_from_vectors():
    assert Matrix._c(
        Vector([1, 2, 3]),
        Vector([4, 5, 6])
    ) == Matrix([
        Vector([1, 4]),
        Vector([2, 5]),
        Vector([3, 6]),
    ])


def test_create_matrix_from_vectors_of_different_size_raise_a_value_error():
    with pytest.raises(ValueError):
        Matrix._c(
            Vector([1]),
            Vector([4, 5])
        )


def accessing_columns():
    pass
