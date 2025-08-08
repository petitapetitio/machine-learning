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

# import random
# sizes = [(10 + i + random.randint(-2, 2)) * 10 for i in range(12)]
# prices = [(10 + size + random.randint(-2, 2)) * 10 for size in sizes]
# dataset = [sizes, prices]
# print(dataset)

dataset = [
    [40, 60, 30, 40, 30, 40, 40, 70, 80, 70, 120, 10],
    [20, 30, 60, 90, 50, 60, 100, 140, 150, 160, 250, 20]
]

current_w = 0
current_b = 0
current_alpha = 0.0001
TRAINING_ITERATIONS = 50


def f_wb(x: float, w: float, b: float) -> float:
    return x * w + b


def cost(dataset: list[list[float]], w: float, b: float) -> float:
    m = len(dataset[0])
    s = 0
    for i in range(m):
        x = dataset[0][i]
        y = dataset[1][i]
        s += (f_wb(x, w, b) - y) * (f_wb(x, w, b) - y)
    s /= 2 * m
    return s


def dw(dataset, w, b):
    m = len(dataset[0])
    s = 0
    for i in range(m):
        x = dataset[0][i]
        y = dataset[1][i]
        s += (f_wb(x, w, b) - y) * x
    s /= m
    return s


def db(dataset, w, b):
    m = len(dataset[0])
    s = 0
    for i in range(m):
        x = dataset[0][i]
        y = dataset[1][i]
        s += (w * x + b - y) * (w * x + b - y) / (2 * m)
    return s


def descent_gradient(w, b, alpha):
    new_w = w - alpha * dw(dataset, w, b)
    new_b = b - alpha * db(dataset, w, b)
    return new_w, new_b


fig, axs = plt.subplots(ncols=2)

# Plot dataset
plot = axs[0]
plot.scatter(dataset[0], dataset[1])
plot.set_xlabel("size")
plot.set_ylabel("price")
X_MIN = 0
X_MAX = 150
plot.set_xlim([X_MIN, X_MAX])
plot.set_ylim([0, 300])

# Plot the model
model_x = [X_MIN, X_MAX]
model_y = [f_wb(X_MIN, current_w, current_b), f_wb(X_MAX, current_w, current_b)]
model_line, = plot.plot(model_x, model_y, 'r-')

# Plot the cost function
plot = axs[1]
ws = [-1 + i * .10 for i in range(100)]
ys = [cost(dataset, current_w, current_b) for current_w in ws]
plot.set_xlabel("w")
plot.set_ylabel("J(w,b)")
plot.plot(ws, ys)
cost_point = plot.scatter(current_w, cost(dataset, current_w, current_b), c='r')


# Animation

def init():
    model_line.set_data([X_MIN, X_MAX], [f_wb(X_MIN, current_w, current_b), f_wb(X_MAX, current_w, current_b)])
    cost_point.set_offsets([[current_w, cost(dataset, current_w, current_b)]])
    return cost_point, model_line


def update(frame):
    global current_w, current_b
    current_w, current_b = descent_gradient(current_w, current_b, current_alpha)
    model_line.set_data([X_MIN, X_MAX], [f_wb(X_MIN, current_w, current_b), f_wb(X_MAX, current_w, current_b)])
    cost_point.set_offsets([[current_w, cost(dataset, current_w, current_b)]])
    return cost_point, model_line


ani = anim.FuncAnimation(
    fig,
    update,
    frames=TRAINING_ITERATIONS + 1,
    init_func=init,
    blit=True,
    interval=30,
    repeat=False
)

plt.show()


