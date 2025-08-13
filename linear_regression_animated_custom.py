"""
The goal is to compute a linear regression of a dataset.
That is, compute w and b such as the cost function is minimized.

1. Create a dataset
2. Plot the dataset
3. Initialize the model with random w and b values
4. Plot the model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import random

from aitk.univariate_linear_regression import UnivariateLinearProblemDataset, UnivariateLinearModel, UnivariateLinearRegression
from numa.vector import Vector

x_vals = Vector.linspace(0, 1, 20)
y_vals = Vector([x + (-.2 + random.random() * .4) for x in x_vals])
dataset = UnivariateLinearProblemDataset.create(x_vals, y_vals)

model = UnivariateLinearModel(0, 0)
regression = UnivariateLinearRegression(n_iterations=100, learning_rate=0.5)


fig, axs = plt.subplots(ncols=2)

# Plot dataset
plot = axs[0]
plot.scatter(list(dataset.X), list(dataset.Y))
plot.set_xlabel("x")
plot.set_ylabel("y")
X_MIN = 0
X_MAX = 1
plot.set_xlim([X_MIN, X_MAX])
plot.set_ylim([0, 1])

# Plot the model
model_x = [X_MIN, X_MAX]
model_y = [model.f(X_MIN), model.f(X_MAX)]
model_line, = plot.plot(model_x, model_y, 'r-')

# Plot the cost function
plot = axs[1]
w_vals = np.linspace(-5, 5, 100)
b_vals = np.linspace(-5, 5, 100)
W, B = np.meshgrid(w_vals, b_vals)
Z = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = UnivariateLinearModel(w=float(W[i, j]), b=float(B[i, j])).cost(dataset)


plot.set_xlabel("w")
plot.set_ylabel("b")
contour = plot.contourf(W, B, Z, levels=50, cmap='viridis')
cbar = fig.colorbar(contour, ax=plot)
cbar.set_label("J(w,b)")
wb_point = plot.scatter(model.w, model.b, color="red")


# Animation

def init():
    model_line.set_data([X_MIN, X_MAX], [model.f(X_MIN), model.f(X_MAX)])
    wb_point.set_offsets([[model.w, model.b]])
    return wb_point, model_line


def update(frame):
    global model
    model = model.descent_gradient(dataset, regression.learning_rate)
    model_line.set_data([X_MIN, X_MAX], [model.f(X_MIN), model.f(X_MAX)])
    wb_point.set_offsets([[model.w, model.b]])
    return wb_point, model_line


ani = anim.FuncAnimation(
    fig,
    update,
    frames=regression.n_iterations + 1,
    init_func=init,
    blit=True,
    interval=60,
    repeat=False
)

plt.show()


