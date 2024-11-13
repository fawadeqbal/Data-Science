import torch


class SimilarityMetrics:
    def __init__(self, data):
        """
        Initializes the SimilarityMetrics class with binary data.

        :param data: A 2D tensor representing binary vectors (each row is a vector)
        """
        self.data = data

    def simple_matching_coefficient(self, x, y):
        """
        Calculate Simple Matching Coefficient between two binary vectors.

        :param x: First binary vector
        :param y: Second binary vector
        :return: Simple Matching Coefficient
        """
        a = torch.sum((x == 1) & (y == 1))  # both 1's
        d = torch.sum((x == 0) & (y == 0))  # both 0's
        b = torch.sum((x == 1) & (y == 0))  # first 1, second 0
        c = torch.sum((x == 0) & (y == 1))  # first 0, second 1
        return (a + d) / (a + b + c + d) if (a + b + c + d) != 0 else 0.0

    def jaccard_similarity(self, x, y):
        """
        Calculate Jaccard similarity between two binary vectors.

        :param x: First binary vector
        :param y: Second binary vector
        :return: Jaccard Similarity
        """
        intersection = torch.sum((x == 1) & (y == 1))  # both 1's
        union = torch.sum((x == 1) | (y == 1))  # either 1 in x or y
        return intersection / union if union != 0 else 0.0

    def compute_similarity_matrix(self):
        """
        Compute the Simple Matching Coefficient and Jaccard similarity for all pairs of data points.
        """
        n = len(self.data)
        smc_matrix = torch.zeros((n, n))
        jaccard_matrix = torch.zeros((n, n))

        for i in range(n):
            for j in range(n):
                smc_matrix[i, j] = self.simple_matching_coefficient(self.data[i], self.data[j])
                jaccard_matrix[i, j] = self.jaccard_similarity(self.data[i], self.data[j])

        return smc_matrix, jaccard_matrix

    def plot_similarity_matrices(self, smc_matrix, jaccard_matrix):
        """
        Plot the similarity matrices as heatmaps.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        # Simple Matching Coefficient Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(smc_matrix.numpy(), annot=True, cmap='coolwarm', cbar=True)
        plt.title("Simple Matching Coefficient Matrix")
        plt.xlabel("Data Points")
        plt.ylabel("Data Points")

        # Jaccard Similarity Matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(jaccard_matrix.numpy(), annot=True, cmap='coolwarm', cbar=True)
        plt.title("Jaccard Similarity Matrix")
        plt.xlabel("Data Points")
        plt.ylabel("Data Points")

        plt.tight_layout()
        plt.show()


# Example Data (3 points with binary attributes)
data = torch.tensor([[1, 0, 1],
                     [1, 1, 0],
                     [0, 1, 1]], dtype=torch.float32)

# Create an instance of SimilarityMetrics
metrics = SimilarityMetrics(data)

# Compute Simple Matching Coefficient and Jaccard similarity matrices
smc_matrix, jaccard_matrix = metrics.compute_similarity_matrix()

# Plot the similarity matrices
metrics.plot_similarity_matrices(smc_matrix, jaccard_matrix)
