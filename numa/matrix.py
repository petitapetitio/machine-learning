from __future__ import annotations

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

    def columns(self) -> list[Vector]:
        return [self.column(i) for i in range(self.n)]

    def rows(self) -> list[Vector]:
        return list(self._rows)

    def row(self, i: int):
        return self._rows[i]



def _are_all_of_same_size(vectors: list[Vector]):
    size = vectors[0].size()
    for col in vectors:
        if col.size() != size:
            return False
    return True