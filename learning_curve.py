import matplotlib.pyplot as plt

from aitk.feature_scaling import z_score_normalization
from aitk.multiple_linear_regression import (
    MultipleLinearProblemDataset,
    MultipleLinearModel,
)
from datasets.dataset_2D_regression import x1, x2, y
from numa.matrix import Matrix
from numa.vector import Vector

dataset = MultipleLinearProblemDataset.create(
    Matrix.with_columns(Vector(z_score_normalization(x1)), Vector(z_score_normalization(x2))),
    Vector(z_score_normalization(y)),
)

NB_ITERATIONS = 2000
model = MultipleLinearModel(Vector.zeros(dataset.nb_features()), 0)


learning_rates = [0.01, 0.05, 0.1, 1]

x = list(range(NB_ITERATIONS))

learning_curves = []
for learning_rate in learning_rates:
    y = [0] * NB_ITERATIONS
    for i in range(NB_ITERATIONS):
        y[i] = model.cost(dataset)
        model = model.descent_gradient(dataset, learning_rate=learning_rate)
    learning_curves.append(y)


fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 1, 1)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("iterations")
ax.set_ylabel("cost")

for alpha, learning_curve in zip(learning_rates, learning_curves):
    ax.plot(x, learning_curve, label=f"alpha = {alpha}")
title = fig.suptitle("Learning curves")

plt.legend()
plt.show()
