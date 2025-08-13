import numpy as np
import matplotlib.pyplot as plt

# --- 1. Create data ---
np.random.seed(0)
X1 = np.random.rand(50)
X2 = np.random.rand(50)
B_true = 5
W1_true = 3
W2_true = 2
y = B_true + W1_true * X1 + W2_true * X2 + np.random.randn(50) * 0.2

# --- 2. Fit model using normal equation ---
# Add bias term (column of ones) to X
X = np.column_stack((np.ones(len(X1)), X1, X2))  # shape: (n_samples, 3)
theta = np.linalg.inv(X.T @ X) @ X.T @ y  # normal equation

B_opt, W1_opt, W2_opt = theta

# --- 3. Cost function (MSE) ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# --- 4. Cost surface for (W1, W2) with B fixed ---
w1_vals = np.linspace(W1_opt - 2, W1_opt + 2, 50)
w2_vals = np.linspace(W2_opt - 2, W2_opt + 2, 50)
W1, W2 = np.meshgrid(w1_vals, w2_vals)

cost_surface = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        y_pred = B_opt + W1[i, j] * X1 + W2[i, j] * X2
        cost_surface[i, j] = mse(y, y_pred)

# --- 5. 1D cost curves ---
def cost_for_w1(w1):
    y_pred = B_opt + w1 * X1 + W2_opt * X2
    return mse(y, y_pred)

def cost_for_w2(w2):
    y_pred = B_opt + W1_opt * X1 + w2 * X2
    return mse(y, y_pred)

def cost_for_b(b):
    y_pred = b + W1_opt * X1 + W2_opt * X2
    return mse(y, y_pred)

w1_curve = np.linspace(W1_opt - 2, W1_opt + 2, 100)
w2_curve = np.linspace(W2_opt - 2, W2_opt + 2, 100)
b_curve = np.linspace(B_opt - 5, B_opt + 5, 100)

cost_w1 = [cost_for_w1(w) for w in w1_curve]
cost_w2 = [cost_for_w2(w) for w in w2_curve]
cost_b  = [cost_for_b(b) for b in b_curve]

# --- 6. Plot ---
fig = plt.figure(figsize=(12, 6))

# Cost surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(W1, W2, cost_surface, cmap='viridis', alpha=0.9)
ax1.set_xlabel('W1')
ax1.set_ylabel('W2')
ax1.set_zlabel('Cost (MSE)')
ax1.set_title(f'Cost Surface (B fixed at {B_opt:.2f})')
ax1.scatter(W1_opt, W2_opt, cost_for_w1(W1_opt), color='red', s=50, label='Optimum')
ax1.legend()

# 1D cost curves
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(w1_curve, cost_w1, label='Cost vs W1')
ax2.axvline(W1_opt, color='blue', linestyle='--')

ax2.plot(w2_curve, cost_w2, label='Cost vs W2')
ax2.axvline(W2_opt, color='orange', linestyle='--')

ax2.plot(b_curve, cost_b, label='Cost vs B')
ax2.axvline(B_opt, color='green', linestyle='--')

ax2.set_xlabel('Parameter value')
ax2.set_ylabel('Cost (MSE)')
ax2.set_title('1D Cost Curves (Others Fixed)')
ax2.legend()

plt.tight_layout()
plt.show()
