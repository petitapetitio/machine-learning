import matplotlib.pyplot as plt
import numpy as np
# from sklearn.linear_model import LinearRegression

# --- 1. Generate example data ---
np.random.seed(0)
X1 = np.random.rand(50)
X2 = np.random.rand(50)
y = 3*X1 + 2*X2 + np.random.randn(50)*0.2

# Combine predictors
X = np.column_stack((X1, X2))

# --- 2. Fit linear regression model ---
# model = LinearRegression()
# model.fit(X, y)

# --- 3. Create a grid for the regression plane ---
x_surf, y_surf = np.meshgrid(
    np.linspace(X1.min(), X1.max(), 20),
    np.linspace(X2.min(), X2.max(), 20)
)
# z_surf = model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf
z_surf = 0.8 + 0.5 * x_surf + 0.1 * y_surf

# --- 4. Plot 3D scatter + regression plane ---
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(X1, X2, y, color='blue', label='Data points')
ax1.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, color='red')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('y')
ax1.set_title('3D Regression Plane')
ax1.legend()

# --- 5. Residuals plot ---
# y_pred = model.predict(X)
# residuals = y - y_pred
#
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.scatter(y_pred, residuals, color='blue')
# ax2.axhline(0, color='red', linestyle='--')
# ax2.set_xlabel('Predicted values')
# ax2.set_ylabel('Residuals')
# ax2.set_title('Residuals vs Predicted')

plt.tight_layout()
plt.show()
