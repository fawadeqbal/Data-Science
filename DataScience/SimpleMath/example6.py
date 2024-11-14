import torch
import matplotlib.pyplot as plt
import numpy as np

# Data: x and y values
x_data = torch.tensor([1, 2, 3], dtype=torch.float32)
y_data = torch.tensor([1, 2, 3], dtype=torch.float32)

# Hypothesis function (with theta_0 = 0)
def hypothesis(theta_1, x):
    return theta_1 * x

# Cost function J(theta)
def cost_function(theta_1, x, y):
    m = len(x)
    h = hypothesis(theta_1, x)
    return (1/(2*m)) * torch.sum((h - y)**2)

# Gradient of J(theta_1) with respect to theta_1
def gradient(theta_1, x, y):
    m = len(x)
    h = hypothesis(theta_1, x)
    grad_1 = torch.sum((h - y) * x) / m
    return grad_1

# Gradient Descent Parameters
learning_rate = 0.01
iterations = 10000
theta_1_initial = torch.tensor([0.0])  # Initial guess for theta_1

# Lists to store values of theta_1 and J(theta) during gradient descent
theta_1_history = []
J_history = []

# Gradient Descent loop
for i in range(iterations):
    grad = gradient(theta_1_initial, x_data, y_data)
    theta_1_initial = theta_1_initial - learning_rate * grad
    J_history.append(cost_function(theta_1_initial, x_data, y_data).item())
    theta_1_history.append(theta_1_initial.item())

# Plotting the cost function J(theta_1) and gradient descent path
plt.figure(figsize=(12, 6))

# Plot Cost function vs theta_1
theta_1_values = np.linspace(-2, 2, 100)
J_values = [cost_function(torch.tensor([theta_1]), x_data, y_data).item() for theta_1 in theta_1_values]

plt.subplot(1, 2, 1)
plt.plot(theta_1_values, J_values, marker='o', color='blue', label='J(θ₁)')
plt.xlabel("θ₁")
plt.ylabel("J(θ₁)")
plt.title("Cost Function J(θ₁) vs θ₁")
plt.grid(True)
plt.legend()

# Plot Linear Regression for each iteration
x_vals = np.linspace(0, 4, 100)  # For plotting the lines
plt.subplot(1, 2, 2)
plt.plot(x_vals, hypothesis(torch.tensor([theta_1_history[-1]]), torch.tensor(x_vals)), label='Fitted Line')

# Plot the actual data points
plt.scatter(x_data, y_data, color='red', label='Data Points (x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final value of theta_1
print(f"Final theta_1 after gradient descent: {theta_1_history[-1]}")
