from __future__ import annotations

import math
from dataclasses import dataclass

from numa.matrix import Matrix
from numa.vector import Vector


@dataclass(frozen=True)
class MultipleLogisticProblemDataset:
    X: Matrix
    y: Vector

    @classmethod
    def create(cls, X: Matrix, y: Vector) -> MultipleLogisticProblemDataset:
        if X.m != y.size():
            raise ValueError
        return MultipleLogisticProblemDataset(X, y)

    def nb_samples(self) -> int:
        return self.y.size()

    def nb_features(self) -> int:
        return self.X.n


@dataclass(frozen=True)
class MultipleLogisticModel:
    w: Vector
    b: float

    def descent_gradient(
        self,
        dataset: MultipleLogisticProblemDataset,
        learning_rate: float,
        lambda_: float = 0,
    ) -> MultipleLogisticModel:
        new_w = self.w - self._dw(dataset, lambda_) * learning_rate
        new_b = self.b - self._db(dataset) * learning_rate
        return MultipleLogisticModel(new_w, new_b)

    def _dw(self, dataset: MultipleLogisticProblemDataset, lambda_: float) -> Vector:
        dw = [0] * dataset.nb_features()
        for j in range(dataset.nb_features()):
            s = 0
            xj = dataset.X.column(j)
            for i in range(dataset.nb_samples()):
                s += (self.f(dataset.X.row(i)) - dataset.y[i]) * xj[i] + (lambda_ * self.w[j])
            s /= dataset.nb_samples()
            dw[j] = s
        return Vector(dw)

    def _db(self, dataset: MultipleLogisticProblemDataset) -> float:
        db = 0
        for x, y in zip(dataset.X.rows(), dataset.y):
            db += self.f(x) - y
        db /= dataset.nb_samples()
        return db

    def f(self, x: Vector) -> float:
        return _sigmoid(x.dot(self.w) + self.b)

    def cost(self, dataset: MultipleLogisticProblemDataset) -> float:
        score = 0
        for x, y in zip(dataset.X.rows(), dataset.y):
            loss = -math.log(self.f(x)) if y >= 0.5 else -math.log(1 - self.f(x))
            score += loss
        score /= 2 * dataset.nb_samples()
        return score


def _sigmoid(z: float, eps=1e-12) -> float:
    s = 1 / (1 + math.exp(-z))
    s = min(max(s, eps), 1 - eps)
    return s


@dataclass(frozen=True)
class MultipleLogisticRegression:
    n_iterations: int
    learning_rate: float

    def fit(self, dataset: MultipleLogisticProblemDataset) -> MultipleLogisticModel:
        model = MultipleLogisticModel(Vector.zeros(dataset.nb_features()), 0)
        for _ in range(self.n_iterations):
            model = model.descent_gradient(dataset, self.learning_rate)

        return model
