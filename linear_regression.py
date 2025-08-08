"""
The goal is to compute a linear regression of a dataset.
That is, compute w and b such as the cost function is minimized.

1. Create a dataset
2. Plot the dataset
3. Initialize the model with random w and b values
4. Plot the model
"""
import matplotlib.pyplot as plt

# import random
# sizes = [i + random.randint(-2, 2) for i in range(12)]
# prices = [2 * size + random.randint(-2, 2) for size in sizes]
# dataset = [sizes, prices]
# print(dataset)

dataset = [
    [40, 60, 30, 40, 30, 40, 40, 70, 80, 70, 120, 10],
    [20, 30, 60, 90, 50, 60, 100, 140, 150, 160, 250, 20]
]


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


current_w = 1
current_b = 0
alpha = 0.0001


fig, axs = plt.subplots(ncols=2)

# plot dataset
plot = axs[0]
plot.scatter(dataset[0], dataset[1])
plot.set_xlabel("size")
plot.set_ylabel("price")
X_MIN = 0
X_MAX = 150
plot.set_xlim([X_MIN, X_MAX])
plot.set_ylim([0, 300])

# plot regression
for i in range(100):
    print(alpha * dw(dataset, current_w, current_b), alpha * db(dataset, current_w, current_b))
    new_w = current_w - alpha * dw(dataset, current_w, current_b)
    new_b = current_b - alpha * db(dataset, current_w, current_b)
    current_w = new_w
    current_b = new_b

model = [[X_MIN, X_MAX], [f_wb(X_MIN, current_w, current_b), f_wb(X_MAX, current_w, current_b)]]
plot.plot(*model)

# Plot the cost function
plot = axs[1]
xs = [-1 + i * .10 for i in range(100)]
ys = [cost(dataset, current_w, current_b) for current_w in xs]
plot.set_xlabel("w")
plot.set_ylabel("J(w,b)")
plot.plot(xs, ys)
plot.scatter(current_w, cost(dataset, current_w, current_b))

plt.show()



