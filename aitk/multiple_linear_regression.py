from __future__ import annotations
from dataclasses import dataclass

from numa.vector import Vector
from numa.matrix import Matrix


@dataclass(frozen=True)
class MultipleLinearProblemDataset:
    X: Matrix
    y: Vector

    @classmethod
    def create(cls, X: Matrix, y: Vector) -> MultipleLinearProblemDataset:
        if X.m != y.size():
            raise ValueError
        return MultipleLinearProblemDataset(X, y)

    def nb_samples(self) -> int:
        return self.y.size()

    def nb_features(self) -> int:
        return self.X.n


@dataclass(frozen=True)
class MultipleLinearModel:
    # score
    # predict

    w: Vector
    b: float

    def descent_gradient(self, dataset: MultipleLinearProblemDataset, learning_rate: float) -> MultipleLinearModel:
        new_w = self.w - self._dw(dataset) * learning_rate
        new_b = self.b - self._db(dataset) * learning_rate
        return MultipleLinearModel(new_w, new_b)

    def _dw(self, dataset: MultipleLinearProblemDataset) -> Vector:
        w = [0] * dataset.nb_features()
        for i in range(dataset.nb_features()):
            s = 0
            xi = dataset.X.column(i)
            for j in range(dataset.nb_samples()):
                s += (self.f(dataset.X.row(j)) - dataset.y[i]) * xi[j]
            s /= dataset.nb_samples()
            w[i] = s
        return Vector(w)

    def _db(self, dataset: MultipleLinearProblemDataset) -> float:
        s = 0
        for x, y in zip(dataset.X.rows(), dataset.y):
            s += self.f(x) - y
        s /= dataset.nb_samples()
        return s

    def f(self, x: Vector) -> float:
        if not isinstance(x, Vector):
            print("ai")
        return sum(x.dot(self.w)) + self.b

    def cost(self, dataset: MultipleLinearProblemDataset) -> float:
        s = 0
        for x, y in zip(dataset.X.rows(), dataset.y):
            prediction_y = self.f(x) - y
            s += prediction_y * prediction_y
        s /= 2 * dataset.nb_samples()
        return s


@dataclass(frozen=True)
class MultipleLinearRegression:
    n_iterations: int
    learning_rate: float

    def fit(self, dataset: MultipleLinearProblemDataset) -> MultipleLinearModel:
        model = MultipleLinearModel(Vector.zeros(dataset.nb_features()), 0)
        for _ in range(self.n_iterations):
            model = model.descent_gradient(dataset, self.learning_rate)

        return model
