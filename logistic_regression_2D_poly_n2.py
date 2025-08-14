import matplotlib.pyplot as plt
from matplotlib import animation

from aitk.feature_scaling import z_score_normalization
from aitk.multiple_logistic_regression import MultipleLogisticProblemDataset, MultipleLogisticModel
from datasets.dataset_2D_classification_poly import x1 as x1_raw
from datasets.dataset_2D_classification_poly import x2 as x2_raw
from datasets.dataset_2D_classification_poly import y as y_raw
from numa.matrix import Matrix
from numa.vector import Vector

x1 = Vector(z_score_normalization(x1_raw))
x2 = Vector(z_score_normalization(x2_raw))
y = Vector(y_raw)

dataset = MultipleLogisticProblemDataset.create(
    X=Matrix.with_columns(
        x1,
        x2,
        x1 ** 2,
        x1.multiply(x2),
        x2 ** 2,
    ),
    y=y
)

n_iterations = 1000
learning_rate = 0.2
model = MultipleLogisticModel(w=Vector([0.01] * dataset.nb_features()), b=0)

i_pos = [i for i, yi in enumerate(y) if yi >= 0.5]
i_neg = [i for i, yi in enumerate(y) if yi < 0.5]

x1_pos = [x1[i] for i in i_pos]
x2_pos = [x2[i] for i in i_pos]

fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.4)
title = fig.suptitle("Multiple Logistic Regression (degree=2)")


ax = fig.add_subplot(1, 3, 1)
ax.set_title("Data and decision boundary")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.scatter(x1_pos, x2_pos, marker="o")

x1_neg = [x1[i] for i in i_neg]
x2_neg = [x2[i] for i in i_neg]
ax.scatter(x1_neg, x2_neg, marker="X")

x1s = Vector.linspace(min(x1), max(x1), 50000)
x2s = [-(model.b + model.w[0] * x1i) / model.w[1] for x1i in x1s]
model_boundary = ax.plot(list(x1s), x2s)[0]

# Plot the learning curve
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title("Learning curve")
ax2.set_xlabel("n iterations")
ax2.set_ylabel("cost")
ax2.set_xlim(0, n_iterations)
ax2.set_ylim(0, 0.4)
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


def boundary_x2(model, x1s):
    # see https://chatgpt.com/c/689d2b65-8e90-8331-bd27-add2f00a2c76
    w = model.w
    b = model.b
    result_upper = []
    result_lower = []

    for x1i in x1s:
        A = w[4]  # coefficient for x2^2
        B = w[1] + w[3] * x1i  # coefficient for x2
        C = b + w[0] * x1i + w[2] * (x1i**2)  # constant term

        if abs(A) < 1e-8:  # Avoid division by zero: behaves like a line
            x2_val = -C / B
            result_upper.append(x2_val)
            result_lower.append(x2_val)
        else:
            disc = B**2 - 4*A*C
            if disc >= 0:
                sqrt_disc = disc**0.5
                x2_up = (-B + sqrt_disc) / (2*A)
                x2_low = (-B - sqrt_disc) / (2*A)
                result_upper.append(x2_up)
                result_lower.append(x2_low)
            else:
                result_upper.append(None)
                result_lower.append(None)
    return result_upper, result_lower


def update(frame):
    global model
    model = model.descent_gradient(dataset, learning_rate)
    x2s_upper, x2s_lower = boundary_x2(model, x1s)
    model_boundary.set_data(list(x1s) + list(x1s), x2s_lower + x2s_upper)

    cost = model.cost(dataset)
    costs_by_iteration.append(cost)
    cost_plot.set_data([list(range(len(costs_by_iteration))), costs_by_iteration])

    params_history[0].append(model.b)
    for i, wi in enumerate(model.w):
        params_history[i+1].append(wi)
    for param_history, param_plot in zip(params_history, param_plots):
        param_plot.set_data([list(range(len(param_history))), param_history])

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
