
"""
TODO
- Vector[bool]
"""
import math
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib import animation

from aitk.univariate_logistic_regression import UnivariateLogisticProblemDataset, UnivariateLogisticModel, sigmoid
from numa.vector import Vector

dataset = UnivariateLogisticProblemDataset(
    Vector([1, 2, 3, 4, 5, 40, 41, 42, 43]),
    Vector([0, 0, 0, 0, 1, 1, 1, 1, 1])
)


def f_reg(model: UnivariateLogisticModel, x):
    return x * model.w + model.b


def index(l: list, condition: Callable):
    for i, x in enumerate(l):
        if condition(x):
            return i

    return 0


def invert_sigmoid(y):
    return math.log(y / (1 - y))


NB_ITERATIONS = 600
learning_rate = 0.1
model = UnivariateLogisticModel(0, 0)

fig = plt.figure(figsize=(10, 5))

title = fig.suptitle("Logistic Regression (iteration = 0)")

ax = fig.add_subplot(1, 1, 1)

ax.scatter(list(dataset.X), list(dataset.Y), color="blue", label="dataset")

x_vals = Vector.linspace(min(dataset.X), max(dataset.X), 10000)
y_vals = [model.f(x) for x in x_vals]
model_plot = ax.plot(list(x_vals), y_vals, color="orange", label="f")[0]
model_boundary_x = x_vals[index(y_vals, lambda y: y >= .5)]
model_boundary = ax.axvline(model_boundary_x, color="orange", linestyle="--", label="f boundary")

# l_vals = [invert_sigmoid(y) for y in y_vals]
l_vals = [f_reg(model, x) for x in x_vals]
line_plot = ax.plot(list(x_vals), l_vals, color="green", label="reg model")[0]
line_boundary_x = x_vals[index(l_vals, lambda y: y >= .5)]
line_boundary = ax.axvline(line_boundary_x, color="green", linestyle="--", label="reg model boundary")


# Animation


def update(frame):
    global model
    model = model.descent_gradient(dataset, learning_rate)

    y_vals = [model.f(x) for x in x_vals]
    model_plot.set_data([list(x_vals), y_vals])
    model_boundary_x = x_vals[index(y_vals, lambda y: y >= .5)]
    model_boundary.set_xdata([model_boundary_x])

    # l_vals = [invert_sigmoid(y) for y in y_vals]
    l_vals = [f_reg(model, x) for x in x_vals]
    line_plot.set_data([list(x_vals), l_vals])
    line_boundary_x = x_vals[index(l_vals, lambda y: y >= .5)]
    line_boundary.set_xdata([line_boundary_x])

    title.set_text(f"Logistic Regression (iteration = {frame + 1})")

    return model_plot, line_plot


ani = animation.FuncAnimation(
    fig,
    update,
    frames=NB_ITERATIONS,
    blit=False,
    interval=60,
    repeat=False
)

plt.legend()
plt.show()