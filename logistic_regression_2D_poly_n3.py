import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from aitk.feature_scaling import z_score_normalization
from aitk.multiple_logistic_regression import MultipleLogisticProblemDataset, MultipleLogisticModel
from datasets.dataset_2D_classification_poly import x1 as x1_raw
from datasets.dataset_2D_classification_poly import x2 as x2_raw
from datasets.dataset_2D_classification_poly import y as y_raw
from numa.matrix import Matrix
from numa.poly import degrees
from numa.vector import Vector

x1 = Vector(z_score_normalization(x1_raw))
x2 = Vector(z_score_normalization(x2_raw))
y = Vector(y_raw)

n = 6
degs = degrees(n)
dataset = MultipleLogisticProblemDataset.create(
    X=Matrix.with_columns(*[(x1**p1).multiply(x2**p2) for p1, p2 in degs]),
    y=y
)

n_iterations = 1000
learning_rate = 0.00001
model = MultipleLogisticModel(w=Vector([0.01] * dataset.nb_features()), b=0)

i_pos = [i for i, yi in enumerate(y) if yi >= 0.5]
i_neg = [i for i, yi in enumerate(y) if yi < 0.5]

fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.4)
title = fig.suptitle(f"Multiple Logistic Regression (degree={n})")


ax = fig.add_subplot(1, 3, 1)
ax.set_title("Data and decision boundary")
ax.set_xlabel("X1")
ax.set_ylabel("X2")

x1_pos = [x1[i] for i in i_pos]
x2_pos = [x2[i] for i in i_pos]
ax.scatter(x1_pos, x2_pos, marker="o")

x1_neg = [x1[i] for i in i_neg]
x2_neg = [x2[i] for i in i_neg]
ax.scatter(x1_neg, x2_neg, marker="X")


def get_contour(m: MultipleLogisticModel, definition: int):
    x1s = Vector.linspace(min(x1), max(x1), definition)
    x2s = Vector.linspace(min(x2), max(x2), definition)
    X1, X2 = np.meshgrid(list(x1s), list(x2s))  # TODO : code my own mesh grid
    Z = np.zeros_like(X1)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            xij1 = X1[i, j]
            xij2 = X2[i, j]
            Z[i, j] = m.f(Vector([(xij1 ** p1) * (xij2 ** p2) for p1, p2 in degs]))
    return X1, X2, Z


contour_definition = 100
X1, X2, Z = get_contour(model, definition=contour_definition)
model_boundary = ax.contour(X1, X2, Z, levels=[0.5], colors='red')

ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title("Learning curve")
ax2.set_xlabel("n iterations")
ax2.set_ylabel("cost")
ax2.set_xlim(0, n_iterations)
ax2.set_ylim(0, 1)
costs_by_iteration = [model.cost(dataset)]
cost_plot = ax2.plot(costs_by_iteration)[0]


ax3 = fig.add_subplot(1, 3, 3)
ax3.set_title("Parameters")
ax3.set_xlim(0, n_iterations)
ax3.set_ylim(-1, 5)
params_history = [[model.b]] + [[wi] for wi in model.w]
param_plots = []

param_plot = ax3.plot(params_history[0], label="b")[0]
param_plots.append(param_plot)
for i, param_history in enumerate(params_history[1:]):
    param_plot = ax3.plot(param_history, label=f"w{i+1}")[0]
    param_plots.append(param_plot)


def update(frame):
    global model, model_boundary
    model = model.descent_gradient(dataset, learning_rate)

    model_boundary.remove()
    X1, X2, Z = get_contour(model, definition=contour_definition)
    model_boundary = ax.contour(X1, X2, Z, levels=[0.5], colors='red')

    cost = model.cost(dataset)
    costs_by_iteration.append(cost)
    cost_plot.set_data(list(range(len(costs_by_iteration))), costs_by_iteration)

    params_history[0].append(model.b)
    for i, wi in enumerate(model.w):
        params_history[i+1].append(wi)
    for param_history, param_plot in zip(params_history, param_plots):
        param_plot.set_data(list(range(len(param_history))), param_history)

    return model_boundary, cost_plot, *param_plots


ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_iterations + 1,
    blit=True,
    interval=60,
    repeat=False
)

plt.legend()
plt.show()
