import torch
import matplotlib.pyplot as plt
import numpy as np

# Data: x and y values
x_data = torch.tensor([1, 2, 3], dtype=torch.float32)
y_data = torch.tensor([1, 2, 3], dtype=torch.float32)


# Hypothesis function (with theta_0 and theta_1)
def hypothesis(theta_0, theta_1, x):
    return theta_0 + theta_1 * x


# Cost function J(theta_0, theta_1)
def cost_function(theta_0, theta_1, x, y):
    m = len(x)
    h = hypothesis(theta_0, theta_1, x)
    return (1 / (2 * m)) * torch.sum((h - y) ** 2)


# Gradient of J(theta_0, theta_1) with respect to theta_0 and theta_1
def gradients(theta_0, theta_1, x, y):
    m = len(x)
    h = hypothesis(theta_0, theta_1, x)
    grad_0 = torch.sum(h - y) / m
    grad_1 = torch.sum((h - y) * x) / m
    return grad_0, grad_1


# Gradient Descent Parameters
learning_rate = 0.1
iterations = 100
theta_0_initial = torch.tensor([0.0])  # Initial guess for theta_0
theta_1_initial = torch.tensor([0.0])  # Initial guess for theta_1

# Lists to store values of theta_0, theta_1, and J(theta) during gradient descent
theta_0_history = []
theta_1_history = []
J_history = []

# Gradient Descent loop
for i in range(iterations):
    grad_0, grad_1 = gradients(theta_0_initial, theta_1_initial, x_data, y_data)
    theta_0_initial = theta_0_initial - learning_rate * grad_0
    theta_1_initial = theta_1_initial - learning_rate * grad_1

    J_history.append(cost_function(theta_0_initial, theta_1_initial, x_data, y_data).item())
    theta_0_history.append(theta_0_initial.item())
    theta_1_history.append(theta_1_initial.item())

# Plotting the cost function J(theta_0, theta_1) as a contour plot
theta_0_values = np.linspace(-2, 2, 100)
theta_1_values = np.linspace(-2, 2, 100)

# Create a grid of theta_0 and theta_1 values
theta_0_grid, theta_1_grid = np.meshgrid(theta_0_values, theta_1_values)
J_values = np.zeros_like(theta_0_grid)

# Compute the cost function for each pair of (theta_0, theta_1)
for i in range(len(theta_0_values)):
    for j in range(len(theta_1_values)):
        J_values[i, j] = cost_function(torch.tensor([theta_0_grid[i, j]]), torch.tensor([theta_1_grid[i, j]]), x_data,
                                       y_data).item()

# Contour plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contour(theta_0_grid, theta_1_grid, J_values, levels=50, cmap='viridis')
plt.colorbar(label='J(θ₀, θ₁)')
plt.xlabel('θ₀')
plt.ylabel('θ₁')
plt.title('Contour Plot of Cost Function J(θ₀, θ₁)')
plt.scatter(theta_0_history[-1], theta_1_history[-1], color='red', label='Optimal (θ₀, θ₁)')
plt.legend()

# Plot the linear regression fit for the final (θ₀, θ₁)
x_vals = np.linspace(0, 4, 100)  # For plotting the lines
plt.subplot(1, 2, 2)
plt.plot(x_vals, hypothesis(theta_0_history[-1], theta_1_history[-1], torch.tensor(x_vals)), label='Fitted Line')

# Plot the actual data points
plt.scatter(x_data, y_data, color='red', label='Data Points (x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final values of theta_0 and theta_1
print(f"Final theta_0: {theta_0_history[-1]}")
print(f"Final theta_1: {theta_1_history[-1]}")
