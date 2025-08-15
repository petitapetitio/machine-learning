import matplotlib.pyplot as plt

from aitk.feature_scaling import z_score_normalization
from aitk.multiple_linear_regression import (
    MultipleLinearProblemDataset,
    MultipleLinearModel,
    MultipleLinearRegression,
)
from datasets.dataset_2D_regression import x1, x2, y
from numa.matrix import Matrix
from numa.vector import Vector

"""
Le dataset non scalé converge rapidement AU DEBUT, mais peine dans les derniers 10%
"""

scaled_dataset = MultipleLinearProblemDataset.create(
    Matrix.with_columns(Vector(z_score_normalization(x1)), Vector(z_score_normalization(x2))),
    Vector(z_score_normalization(y)),
)
unscaled_dataset = MultipleLinearProblemDataset.create(Matrix.with_columns(Vector(x1), Vector(x2)), Vector(y))

NB_ITERATIONS = 2000
scaled_model = MultipleLinearModel(Vector.zeros(scaled_dataset.nb_features()), 0)
unscaled_model = MultipleLinearModel(Vector.zeros(unscaled_dataset.nb_features()), 0)

scaled_learning_rate = 0.4
unscaled_learning_rate = 0.00004

x = list(range(NB_ITERATIONS))
learning_curve_scaled = [0] * NB_ITERATIONS
learning_curve_unscaled = [0] * NB_ITERATIONS
for i in range(NB_ITERATIONS):
    learning_curve_scaled[i] = scaled_model.cost(scaled_dataset)
    learning_curve_unscaled[i] = unscaled_model.cost(unscaled_dataset)
    scaled_model = scaled_model.descent_gradient(scaled_dataset, learning_rate=scaled_learning_rate)
    unscaled_model = unscaled_model.descent_gradient(unscaled_dataset, learning_rate=unscaled_learning_rate)


fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 1, 1)

ax1.axhline(0, color="grey", linewidth=1)
ax1.set_xlabel("iterations")
ax1.set_ylabel("cost")
ax1.plot(x, learning_curve_scaled, label=f"scaled", color="blue")

ax2 = ax1.twinx()
ax2.plot(x, learning_curve_unscaled, label=f"unscaled", color="orange")
ax2.set_ylabel("cost (of the unscaled)")
ax2.set_ylim(0, 1e8)

fig.suptitle("Learning curves (scaled vs unscaled datasets)")
plt.legend()
plt.show()
