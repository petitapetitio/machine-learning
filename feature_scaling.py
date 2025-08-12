from aitk.feature_scaling import divide_by_max, mean_normalization, z_score_normalization
from datasets.dataset_1D_regression import x
import matplotlib.pyplot as plt

x = [xi * 100 for xi in x]

fig, axs = plt.subplots(1, 4)
fig.set_size_inches(14, 6)

axs[0].plot(x)
axs[0].set_title("Original")

axs[1].plot(divide_by_max(x))
axs[1].set_title("Divide by max")

axs[2].plot(mean_normalization(x))
axs[2].set_title("Mean normlization")

axs[3].plot(z_score_normalization(x))
axs[3].set_title("Z-score")

plt.subplots_adjust(wspace=0.4)
fig.suptitle("FEATURE SCALING")
plt.show()



