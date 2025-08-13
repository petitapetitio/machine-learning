from __future__ import annotations

from numbers import Number
from typing import Iterable


class Vector(Iterable):
    def __init__(self, values: list):
        self._elements = list(values)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self._elements == other._elements
        return False

    def __repr__(self):
        return f"Vector([{self._elements}])"

    def __add__(self, other):
        if isinstance(other, Vector) and self.size() == other.size():
            return Vector([x + y for x, y in zip(self._elements, other._elements)])

        raise ValueError

    def __sub__(self, other):
        if isinstance(other, Vector) and self.size() == other.size():
            return Vector([x - y for x, y in zip(self._elements, other._elements)])
        raise ValueError

    def __mul__(self, other):
        if isinstance(other, Number) and not isinstance(other, bool):
            return Vector([x * other for x in self._elements])
        raise ValueError

    def size(self) -> int:
        return len(self._elements)

    def dot(self, other: Vector) -> Vector:
        if self.size() == other.size():
            return Vector([x * y for x, y in zip(self._elements, other._elements)])
        raise ValueError

    def __iter__(self):
        return iter(self._elements)

    def __getitem__(self, item):
        return self._elements[item]

    @classmethod
    def linspace(cls, lo: float, hi: float, n: int):
        delta = (hi - lo) / (n - 1)
        return Vector([lo + i * delta for i in range(n)])

    @classmethod
    def zeros(cls, n):
        if n < 0:
            raise ValueError

        return Vector([0] * n)
