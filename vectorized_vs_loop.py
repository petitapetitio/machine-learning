import random
import time
from timeit import timeit

import numpy as np
import matplotlib.pyplot as plt


def dot_scalar(v1: list[float], v2: list[float]) -> list[float]:
    return [a * b for a, b in zip(v1, v2)]


N = 100
repeat = 10
scalar_times = []
vectorized_times = []
xs = []
for n in range(1, N):
    v1 = [random.random() for _ in range(n)]
    v2 = [random.random() for _ in range(n)]
    duration = timeit(lambda: dot_scalar(v1, v2), number=repeat)
    scalar_times.append(duration)

    v1 = np.random.random(n)
    v2 = np.random.random(n)
    t = time.time()
    duration = timeit(lambda: np.dot(v1, v2), number=repeat)
    vectorized_times.append(duration)

    xs.append(n)

plt.title("Cost of dot product (loop vs vectorization)")
plt.xlabel("Size of the vectors")
plt.ylabel("Time (in ms)")
plt.plot(xs, scalar_times, label="loop")
plt.plot(xs, vectorized_times, label="vectorized")
plt.legend()
plt.show()
