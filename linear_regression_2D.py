import matplotlib.pyplot as plt
import matplotlib.animation as anim

import numpy as np
from matplotlib.ticker import MaxNLocator

from aitk.feature_scaling import z_score_normalization
from aitk.multiple_linear_regression import MultipleLinearProblemDataset, MultipleLinearModel, MultipleLinearRegression
from datasets.dataset_2D_regression import x1, x2, y
from numa.vector import Vector
from numa.matrix import Matrix

dataset = MultipleLinearProblemDataset.create(
    Matrix.with_columns(Vector(z_score_normalization(x1)), Vector(z_score_normalization(x2))),
    Vector(z_score_normalization(y))
)

model = MultipleLinearModel(Vector.zeros(dataset.nb_features()), 0)
regression = MultipleLinearRegression(n_iterations=1000, learning_rate=0.1)


fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.4)
title = fig.suptitle("Multiple Linear Regression (iteration = 0)")


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

x_bounds = 0.0008
X_MIN = -x_bounds
X_MAX = x_bounds
w1_vals = np.linspace(X_MIN, X_MAX, 100)
w1_cost = [MultipleLinearModel(w=Vector([w1i, model.w[1]]), b=model.b).cost(dataset) for w1i in w1_vals]
w1_plot = ax2.plot(w1_vals, w1_cost, label='w1', color="blue")[0]
w1_point = ax2.scatter(model.w[0], model.cost(dataset), color='blue')

w2_vals = np.linspace(X_MIN, X_MAX, 100)
w2_cost = [MultipleLinearModel(w=Vector([model.w[0], w2i]), b=model.b).cost(dataset) for w2i in w2_vals]
w2_plot = ax2.plot(w2_vals, w2_cost, label='w2', color='orange')[0]
w2_point = ax2.scatter(model.w[1], model.cost(dataset), color='orange')

b_vals = np.linspace(X_MIN, X_MAX, 100)
b_cost = [MultipleLinearModel(w=model.w, b=b).cost(dataset) for b in b_vals]
b_plot = ax2.plot(b_vals, b_cost, label='b', color='green')[0]
b_point = ax2.scatter(model.b, model.cost(dataset), color='green')

ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax2.set_ylim(-1e-8, 0.00000003)
ax2.set_xlabel("parameter's value")
ax2.set_ylabel("cost")
ax2.set_title("Model's parameters cost functions")
ax2.legend()


# Animation

def update(frame):
    global model, model_surface, x_bounds
    model = model.descent_gradient(dataset, regression.learning_rate)

    model_surface.remove()
    z_surf = model.b + model.w[0] * x_surf + model.w[1] * y_surf
    model_surface = ax1.plot_surface(x_surf, y_surf, z_surf, alpha=0.2, color='orange')

    cost = model.cost(dataset)
    w1_cost = [MultipleLinearModel(w=Vector([w1i, model.w[1]]), b=model.b).cost(dataset) for w1i in w1_vals]
    w1_plot.set_data([w1_vals, w1_cost])
    w1_point.set_offsets([[model.w[0], cost]])

    w2_cost = [MultipleLinearModel(w=Vector([model.w[0], w2i]), b=model.b).cost(dataset) for w2i in w2_vals]
    w2_plot.set_data([w2_vals, w2_cost])
    w2_point.set_offsets([[model.w[1], cost]])

    b_cost = [MultipleLinearModel(w=model.w, b=b).cost(dataset) for b in b_vals]
    b_plot.set_data([b_vals, b_cost])
    b_point.set_offsets([[model.b, cost]])

    title.set_text(f"Multiple Linear Regression (iteration = {frame+1})")

    return model_surface, w1_plot, w1_point, w2_plot, w2_point, b_plot, b_point


ani = anim.FuncAnimation(
    fig,
    update,
    frames=regression.n_iterations + 1,
    blit=False,
    interval=60,
    repeat=False
)

plt.show()


