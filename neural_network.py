from aitk.univariate_logistic_regression import sigmoid
from numa.vector import Vector

"""
Ce qui est intéressant avec cette implémentation, c'est qu'on voit que la logique derrière un NN est simple. 

La notation matricielle n'est qu'une façon de réécrire ces équations et optimiser les calculs grace à la vectorisation.
"""


class Unit:
    def __init__(self, w: Vector, b: float):
        self._w = w
        self._b = b

    def activation(self, input_: Vector) -> float:
        return sigmoid(self._w.dot(input_) + self._b)


class Layer:
    def __init__(self, units: list[Unit]):
        self._units = units

    def activation(self, input_: Vector) -> Vector:
        return Vector([unit.activation(input_) for unit in self._units])


class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self._layers = layers

    def predict(self, x: Vector):
        a = x
        for layer in self._layers:
            a = layer.activation(a)
        return a


model = NeuralNetwork(
    [
        Layer(
            units=[
                Unit(Vector([-8.93, -0.1]), -9.82),
                Unit(Vector([0.29, -7.32]), -9.28),
                Unit(Vector([12.9, 10.81]), 0.96),
            ]
        ),
        Layer(units=[Unit(Vector([-31.18, -27.59, -32.56]), 15.41)]),
    ]
)


def remap(x, lo, hi, new_lo, new_hi):
    return new_lo + ((x - lo) / (hi - lo)) * (new_hi - new_lo)


# for i in range(1000):
#     for j in range(1000):
#         sample = Vector([remap(i, 0, 1000, -1, 1), remap(j, 0, 1000, -1, 1)])
#         y = model.predict(sample)
#         if y[0] > 0.5:
#             print("00", sample)
#         else:
#             print("XX", sample)

positive = Vector([-0.1959, 0.1259])
negative = Vector([-0.1939, 0.3799])
print(f"{model.predict(positive)=}")
print(f"{model.predict(negative)=}")
