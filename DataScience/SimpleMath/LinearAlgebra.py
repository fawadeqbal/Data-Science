import torch

# Define a sample matrix
A = torch.tensor([[2.0, 3.0,4.0], [4.0, 5.0,5.0],[4.0, 5.0,5.0]])

# Matrix operations
B = torch.tensor([[1.0, 1.0,8.0], [4.0,2.0, 2.0],[3.5,5.4,5.6]])

# Basic Operations
addition = A + B
subtraction = A - B
elementwise_multiplication = A * B
scalar_multiplication = A * 3

# Matrix Inversion
inverse_A = torch.linalg.inv(A)

# Eigenvalue Decomposition
eigenvalues, eigenvectors = torch.linalg.eig(A)

# Singular Value Decomposition
U, S, V = torch.svd(A)

# Solving a linear system
b = torch.tensor([1.0, 2.0,3.0])
solution = torch.linalg.solve(A, b)

# Norms
frobenius_norm = torch.norm(A, 'fro')
l2_norm = torch.norm(A, 0)

# Rank of a matrix
rank = torch.linalg.matrix_rank(A)

# Trace
trace = torch.trace(A)

# Output results
print(f"Addition: \n{addition}")
print(f"Subtraction: \n{subtraction}")
print(f"Element-wise multiplication: \n{elementwise_multiplication}")
print(f"Scalar multiplication: \n{scalar_multiplication}")
print(f"Inverse of A: \n{inverse_A}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors: {eigenvectors}")
print(f"SVD U: \n{U}, S: \n{S}, V: \n{V}")
print(f"Solution to Ax = b: {solution}")
print(f"Frobenius Norm: {frobenius_norm}")
print(f"L2 Norm: {l2_norm}")
print(f"Matrix Rank: {rank}")
print(f"Matrix Trace: {trace}")
