"""
Charger le dataset
Dessiner le dataset en mode scatter (x1, x2) (style différent selon True/False, séparer les données avant)
Coder la regression logistique multiple
Afficher les coefficients
Dessiner le modèle

Créer version polynomiale
Ajouter régularisation
"""
import matplotlib.pyplot as plt
from matplotlib import animation

from aitk.feature_scaling import z_score_normalization
from aitk.multiple_logistic_regression import MultipleLogisticProblemDataset, MultipleLogisticModel
from datasets.dataset_2D_classification import x1, x2, y
from numa.matrix import Matrix
from numa.vector import Vector

"""
Cet exemple démontre bien l'importance de normaliser les données. 
Il est très difficile de faire converger le modèle avec les features brutes.
"""

x1 = z_score_normalization(x1)
x2 = z_score_normalization(x2)

dataset = MultipleLogisticProblemDataset.create(
    X=Matrix.with_columns(
        Vector(x1), Vector(x2)),
    y=Vector(y)
)

n_iterations = 1000
learning_rate = 5
# model = MultipleLogisticModel(w=Vector([0.03070169046435127, 0.02275950116778541]), b=-2.8494625416534785)
model = MultipleLogisticModel(w=Vector([0.01, 0.01]), b=0)

i_pos = [i for i, yi in enumerate(y) if yi >= 0.5]
i_neg = [i for i, yi in enumerate(y) if yi < 0.5]

x1_pos = [x1[i] for i in i_pos]
x2_pos = [x2[i] for i in i_pos]

fig = plt.figure(figsize=(6, 6))
plt.subplots_adjust(wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.scatter(x1_pos, x2_pos, marker="o")

x1_neg = [x1[i] for i in i_neg]
x2_neg = [x2[i] for i in i_neg]
ax.scatter(x1_neg, x2_neg, marker="X")

x1s = Vector.linspace(min(x1), max(x1), 100)
x2s = [-(model.b + model.w[0] * x1i) / model.w[1] for x1i in x1s]
model_boundary = ax.plot(list(x1s), x2s)[0]

# Plot the learning curve
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlim(0, n_iterations)
ax2.set_ylim(0, 0.4)
costs_by_iteration = [model.cost(dataset)]
cost_plot = ax2.plot(costs_by_iteration)[0]

# y = [0] * n_iterations
# m = MultipleLogisticModel(Vector.zeros(2), 0)
# for i in range(n_iterations):
#     y[i] = m.cost(dataset)
#     m = m.descent_gradient(dataset, learning_rate=learning_rate)
# ax2.plot(list(range(n_iterations)), y)


def update(frame):
    global model
    model = model.descent_gradient(dataset, learning_rate)
    x2s = [-(model.b + model.w[0] * x1i) / model.w[1] for x1i in x1s]
    model_boundary.set_data(list(x1s), x2s)

    costs_by_iteration.append(model.cost(dataset))
    cost_plot.set_data([list(range(len(costs_by_iteration))), costs_by_iteration])

    return model_boundary, cost_plot


ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_iterations + 1,
    blit=True,
    interval=60,
    repeat=False
)

plt.show()
