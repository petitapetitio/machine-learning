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

from typing import Generator

import pytest

from numa.vector import Vector


class Matrix:
    def __init__(self, rows: list[Vector]):
        self._rows = rows
        self.m = len(rows)
        self.n = 0 if len(rows) == 0 else rows[0].size()

    @classmethod
    def create(cls, rows: list[Vector]) -> Matrix:
        if not _are_all_of_same_size(rows):
            raise ValueError
        return Matrix(rows)

    @classmethod
    def with_columns(cls, *columns: Vector) -> Matrix:
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

    def column(self, i: int):
        if i < 0 or i >= self.n:
            raise IndexError(f"{i} out of range [0, {self.n - 1}")
        return Vector([r[i] for r in self._rows])

    def columns(self) -> Generator[Vector, None, None]:
        for i in range(self.n):
            yield self.column(i)


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
    assert Matrix.with_columns(
        Vector([1, 2, 3]),
        Vector([4, 5, 6])
    ) == Matrix([
        Vector([1, 4]),
        Vector([2, 5]),
        Vector([3, 6]),
    ])


def test_create_matrix_from_vectors_of_different_size_raise_a_value_error():
    with pytest.raises(ValueError):
        Matrix.with_columns(
            Vector([1]),
            Vector([4, 5])
        )


def test_column_are_indexed_from_zero():
    assert Matrix.with_columns(Vector([1, 2, 3])).column(0) == Vector([1, 2, 3])


def test_accessing_column_by_invalid_index_raise_index_error():
    with pytest.raises(IndexError):
        Matrix.with_columns(Vector([1, 2, 3])).column(1)


def test_accessing_columns():
    assert list(Matrix.with_columns(
        Vector([1, 2]),
        Vector([4, 5])
    ).columns()) == [
        Vector([1, 2]),
        Vector([4, 5]),
    ]