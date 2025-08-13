"""
The goal is to compute a linear regression of a dataset.
That is, compute w and b such as the cost function is minimized.

1. Create a dataset
2. Plot the dataset
3. Initialize the model with random w and b values
4. Plot the model
"""

import matplotlib.pyplot as plt
import matplotlib.animation as anim

import numpy as np

from aitk.multiple_linear_regression import MultipleLinearProblemDataset, MultipleLinearModel, MultipleLinearRegression
from datasets.dataset_2D_regression import x1, x2, y
from numa.vector import Vector
from numa.matrix import Matrix

# TODO : afficher le nombre d'itérations

dataset = MultipleLinearProblemDataset.create(
    Matrix.with_columns(Vector(x1), Vector(x2)),
    Vector(y)
)

model = MultipleLinearModel(Vector.zeros(dataset.nb_features()), 0)
regression = MultipleLinearRegression(n_iterations=5000, learning_rate=0.0005)


fig = plt.figure(figsize=(12, 6))


# Plot dataset and the model
ax1 = fig.add_subplot(1, 2, 1, projection="3d")

ax1.scatter(list(dataset.X.column(0)), list(dataset.X.column(1)), list(dataset.y))

x_surf, y_surf = np.meshgrid(
    list(dataset.X.column(0)),
    list(dataset.X.column(1)),
)
z_surf = model.b + model.w[0] * x_surf + model.w[1] * y_surf
model_surface = ax1.plot_surface(x_surf, y_surf, z_surf, alpha=0.2, color='orange')

ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("y")
ax1.set_title('3D Regression Plane')

# Plot the cost function
ax2 = fig.add_subplot(1, 2, 2)

w1_vals = np.linspace(-5000, 5000, 100)
w1_cost = [MultipleLinearModel(w=Vector([w1i, model.w[1]]), b=model.b).cost(dataset) for w1i in w1_vals]
ax2.plot(w1_vals, w1_cost, label='w1', color="blue")
ax2.scatter(model.w[0], model.cost(dataset), color='blue')

w2_vals = np.linspace(-5000, 5000, 100)
w2_cost = [MultipleLinearModel(w=Vector([model.w[0], w2i]), b=model.b).cost(dataset) for w2i in w2_vals]
ax2.plot(w2_vals, w2_cost, label='w2')
ax2.scatter(model.w[1], model.cost(dataset), color='orange')

b_vals = np.linspace(-5000, 5000, 100)
b_cost = [MultipleLinearModel(w=model.w, b=b).cost(dataset) for b in b_vals]
ax2.plot(b_vals, b_cost, label='b', color='green')
ax2.scatter(model.b, model.cost(dataset), color='green')

ax2.legend()


# Animation

def init():
    global model_surface
    model_surface.remove()
    z_surf = model.b + model.w[0] * x_surf + model.w[1] * y_surf
    model_surface = ax1.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, color='orange')
    return model_surface,


def update(frame):
    global model, model_surface
    model = model.descent_gradient(dataset, regression.learning_rate)
    model_surface.remove()
    z_surf = model.b + model.w[0] * x_surf + model.w[1] * y_surf
    model_surface = ax1.plot_surface(x_surf, y_surf, z_surf, alpha=0.2, color='orange')
    return model_surface,


ani = anim.FuncAnimation(
    fig,
    update,
    frames=regression.n_iterations + 1,
    init_func=init,
    blit=False,
    interval=60,
    repeat=False
)

plt.show()


