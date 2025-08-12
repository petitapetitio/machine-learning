from __future__ import annotations
from dataclasses import dataclass

from numa.vector import Vector


# TODO : implement MultipleLinearRegression


@dataclass(frozen=True)
class UnivariateLinearProblemDataset:
    X: Vector
    Y: Vector

    @classmethod
    def create(cls, X: Vector, Y: Vector) -> UnivariateLinearProblemDataset:
        if X.size() != Y.size():
            raise ValueError
        return UnivariateLinearProblemDataset(X, Y)

    def size(self) -> int:
        return self.X.size()


@dataclass(frozen=True)
class UnivariateLinearModel:
    # score
    # predict

    w: float
    b: float

    def descent_gradient(self, dataset: UnivariateLinearProblemDataset, learning_rate: float) -> UnivariateLinearModel:
        new_w = self.w - learning_rate * self._dw(dataset)
        new_b = self.b - learning_rate * self._db(dataset)
        return UnivariateLinearModel(new_w, new_b)

    def _dw(self, dataset: UnivariateLinearProblemDataset) -> float:
        s = 0
        for x, y in zip(dataset.X, dataset.Y):
            s += (self.f(x) - y) * x
        s /= dataset.size()
        return s

    def _db(self, dataset: UnivariateLinearProblemDataset) -> float:
        s = 0
        for x, y in zip(dataset.X, dataset.Y):
            s += self.f(x) - y
        s /= dataset.size()
        return s

    def f(self, x: float) -> float:
        return x * self.w + self.b

    def cost(self, dataset: UnivariateLinearProblemDataset) -> float:
        s = 0
        for x, y in zip(dataset.X, dataset.Y):
            prediction_y = self.f(x) - y
            s += prediction_y * prediction_y
        s /= 2 * dataset.size()
        return s



@dataclass(frozen=True)
class UnivariateLinearRegression:
    n_iterations: int
    learning_rate: float

    def fit(self, dataset: UnivariateLinearProblemDataset) -> UnivariateLinearModel:
        model = UnivariateLinearModel(0, 0)
        for _ in range(self.n_iterations):
            model = model.descent_gradient(dataset, self.learning_rate)

        return model
