import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD:
    def __init__(self, x, y, learning_rate=0.1, iterations=10):
        # Initial data and hyperparameters
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.iterations = iterations
        # Initialize parameters
        self.theta0 = 0.0  # intercept
        self.theta1 = 0.0  # slope
        # Track progress for plotting
        self.theta0_vals = [self.theta0]
        self.theta1_vals = [self.theta1]
        self.cost_vals = []

    def compute_cost(self):
        """Compute the Mean Squared Error cost function."""
        m = len(self.y)
        predictions = self.theta0 + self.theta1 * self.x
        cost = (1 / (2 * m)) * np.sum((predictions - self.y) ** 2)
        return cost

    def gradient_descent(self):
        """Run the gradient descent optimization process."""
        for _ in range(self.iterations):
            m = len(self.y)
            predictions = self.theta0 + self.theta1 * self.x
            # Calculate gradients
            d_theta0 = -(1 / m) * np.sum(self.y - predictions)
            d_theta1 = -(1 / m) * np.sum((self.y - predictions) * self.x)
            # Update parameters
            self.theta0 -= self.learning_rate * d_theta0
            self.theta1 -= self.learning_rate * d_theta1
            # Store values for tracking
            self.theta0_vals.append(self.theta0)
            self.theta1_vals.append(self.theta1)
            self.cost_vals.append(self.compute_cost())

    def plot_cost(self):
        """Plot the cost reduction over iterations."""
        plt.plot(range(1, len(self.cost_vals) + 1), self.cost_vals, marker='o')
        plt.title('Cost Reduction Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (Mean Squared Error)')
        plt.grid(True)

    def plot_progression(self):
        """Plot the line evolution over iterations."""
        plt.scatter(self.x, self.y, color='red', label="Data Points")
        x_vals = np.linspace(0, 4, 100)
        for i in range(0, len(self.theta0_vals), 2):  # Plot every 2nd update
            y_vals = self.theta0_vals[i] + self.theta1_vals[i] * x_vals
            plt.plot(x_vals, y_vals, label=f'Iteration {i}', alpha=0.5)
        plt.title('Gradient Descent Progression')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc="upper left", fontsize='small')
        plt.grid(True)

    def plot(self):
        """Plot both the cost and progression of gradient descent."""
        plt.figure(figsize=(12, 6))
        # Plot the cost vs. iteration
        plt.subplot(1, 2, 1)
        self.plot_cost()
        # Plot the line progression
        plt.subplot(1, 2, 2)
        self.plot_progression()
        plt.tight_layout()
        plt.show()


x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

model = LinearRegressionGD(x, y, learning_rate=0.1, iterations=10)

model.gradient_descent()
model.plot()
