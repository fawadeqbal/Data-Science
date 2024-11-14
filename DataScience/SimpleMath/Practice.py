import torch
import matplotlib.pyplot as plt

# Define the quadratic function f(x) = ax^2 + bx + c
def quadratic_function(x, a=1, b=0, c=0):
    return a * x**2 + b * x + c

# Derivative of the quadratic function f'(x) = 2ax + b
def derivative(f, x, a=1, b=0, c=0):
    x_tensor = torch.tensor(x, requires_grad=True)
    y = f(x_tensor, a, b, c)
    y.backward()  # Compute the derivative
    return x_tensor.grad.item()

# Approximate the integral of the derivative using the trapezoidal rule
def integrate_derivative(derivatives, x_values):
    integral_values = torch.cumsum((derivatives[1:] + derivatives[:-1]) * (x_values[1] - x_values[0]) / 2, dim=0)
    integral_values = torch.cat((torch.tensor([0]), integral_values))  # Include the constant of integration as 0
    return integral_values

# Set coefficients for the quadratic function
a, b, c = 1, -2, 1  # Example: f(x) = x^2 - 2x + 1

# Generate x values, and calculate function values, derivatives, and approximate integral
x_values = torch.linspace(-10, 10, steps=100)
y_values = quadratic_function(x_values, a, b, c)

# Calculate the derivative at each point
y_derivatives = [derivative(quadratic_function, x, a, b, c) for x in x_values]

# Approximate the integral of the derivative
integral_values = integrate_derivative(torch.tensor(y_derivatives), x_values)+120

# Plotting f(x), f'(x), and the integral
plt.figure(figsize=(12, 8))

# Plot f(x) = ax^2 + bx + c
plt.plot(x_values.numpy(), y_values.numpy(), label="f(x) = x^2 - 2x + 1", color="blue")

# Plot the derivative f'(x) = 2x - 2
plt.plot(x_values.numpy(), y_derivatives, label="f'(x) = 2x - 2", color="red", linestyle='--')

# Plot the integral of f'(x), which should return f(x)
plt.plot(x_values.numpy(), integral_values.numpy(), label="Integral of f'(x)", color="green", linestyle=':')

# Graph labels and legend
plt.xlabel("x")
plt.ylabel("Function, Derivative, and Integral")
plt.title("Quadratic Function, Its Derivative, and Integral")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
