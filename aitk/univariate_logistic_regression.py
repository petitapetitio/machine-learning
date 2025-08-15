from __future__ import annotations

import math
from dataclasses import dataclass

from numa.vector import Vector


@dataclass(frozen=True)
class UnivariateLogisticProblemDataset:
    X: Vector
    Y: Vector

    @classmethod
    def create(cls, X: Vector, Y: Vector) -> UnivariateLogisticProblemDataset:
        if X.size() != Y.size():
            raise ValueError
        return UnivariateLogisticProblemDataset(X, Y)

    def size(self) -> int:
        return self.X.size()


@dataclass(frozen=True)
class UnivariateLogisticModel:
    w: float
    b: float

    def descent_gradient(
        self, dataset: UnivariateLogisticProblemDataset, learning_rate: float
    ) -> UnivariateLogisticModel:
        new_w = self.w - learning_rate * self._dw(dataset)
        new_b = self.b - learning_rate * self._db(dataset)
        return UnivariateLogisticModel(new_w, new_b)

    def _dw(self, dataset: UnivariateLogisticProblemDataset) -> float:
        s = 0
        for x, y in zip(dataset.X, dataset.Y):
            s += (self.f(x) - y) * x
        s /= dataset.size()
        return s

    def _db(self, dataset: UnivariateLogisticProblemDataset) -> float:
        s = 0
        for x, y in zip(dataset.X, dataset.Y):
            s += self.f(x) - y
        s /= dataset.size()
        return s

    def f(self, x: float) -> float:
        return sigmoid(x * self.w + self.b)

    def cost(self, dataset: UnivariateLogisticProblemDataset) -> float:
        score = 0
        for x, y in zip(dataset.X, dataset.Y):
            loss = (-y * math.log(self.f(x))) - (1 - y) * math.log(1 - self.f(x))
            score += loss
        score /= 2 * dataset.size()
        return score


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


@dataclass(frozen=True)
class UnivariateLogisticRegression:
    n_iterations: int
    learning_rate: float

    def fit(self, dataset: UnivariateLogisticProblemDataset) -> UnivariateLogisticModel:
        model = UnivariateLogisticModel(0, 0)
        for _ in range(self.n_iterations):
            model = model.descent_gradient(dataset, self.learning_rate)

        return model
