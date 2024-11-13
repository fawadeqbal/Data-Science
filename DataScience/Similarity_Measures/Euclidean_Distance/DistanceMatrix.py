import torch
import matplotlib.pyplot as plt
import seaborn as sns


class DistanceMatrix:
    def __init__(self, data):
        """
        Initializes the DistanceMatrix class with given data.

        :param data: A 2D tensor representing data points (each row is a point)
        """
        self.data = data

    def manhattan_distance(self, x, y):
        """
        Calculate Manhattan (L1) distance between two points.
        """
        return torch.sum(torch.abs(x - y))

    def euclidean_distance(self, x, y):
        """
        Calculate Euclidean (L2) distance between two points.
        """
        return torch.sqrt(torch.sum((x - y) ** 2))

    def chebyshev_distance(self, x, y):
        """
        Calculate Chebyshev (L∞) distance between two points.
        """
        return torch.max(torch.abs(x - y))

    def calculate_distances(self, distance_fn):
        """
        Calculate the distance matrix using the provided distance function.

        :param distance_fn: A function that computes the distance between two points.
        :return: A tensor containing the distance matrix.
        """
        n = len(self.data)
        dist_matrix = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = distance_fn(self.data[i], self.data[j])
        return dist_matrix

    def plot_distance_matrix(self, dist_matrix, title):
        """
        Plot the distance matrix as a heatmap.

        :param dist_matrix: The distance matrix to be plotted.
        :param title: Title for the heatmap.
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(dist_matrix.numpy(), annot=True, cmap='coolwarm', cbar=True)
        plt.title(title)
        plt.xlabel("Data Points")
        plt.ylabel("Data Points")
        plt.show()

    def plot_manhattan_matrix(self):
        """Compute and plot the Manhattan distance matrix."""
        manhattan_distances = self.calculate_distances(self.manhattan_distance)
        self.plot_distance_matrix(manhattan_distances, "Manhattan Distance Matrix")

    def plot_euclidean_matrix(self):
        """Compute and plot the Euclidean distance matrix."""
        euclidean_distances = self.calculate_distances(self.euclidean_distance)
        self.plot_distance_matrix(euclidean_distances, "Euclidean Distance Matrix")

    def plot_chebyshev_matrix(self):
        """Compute and plot the Chebyshev distance matrix."""
        chebyshev_distances = self.calculate_distances(self.chebyshev_distance)
        self.plot_distance_matrix(chebyshev_distances, "Chebyshev Distance Matrix")


# Example Data (3 points with 3 attributes)
data = torch.tensor([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]])

# Create an instance of DistanceMatrix
distance_matrix = DistanceMatrix(data)

distance_matrix.plot_manhattan_matrix()

distance_matrix.plot_euclidean_matrix()

distance_matrix.plot_chebyshev_matrix()
