import torch


# Basic Mathematical Operations

def add_tensors(a, b):
    """Adds two tensors element-wise."""
    return a + b


def subtract_tensors(a, b):
    """Subtracts tensor b from tensor a element-wise."""
    return a - b


def multiply_tensors(a, b):
    """Multiplies two tensors element-wise."""
    return a * b


def divide_tensors(a, b):
    """Divides tensor a by tensor b element-wise."""
    return a / b


def scalar_multiply(tensor, scalar):
    """Multiplies a tensor by a scalar value."""
    return tensor * scalar


def power_tensors(a, exponent):
    """Raises tensor a to the given exponent element-wise."""
    return a ** exponent


def sqrt_tensors(tensor):
    """Computes the square root of each element in the tensor."""
    return torch.sqrt(tensor.float())


# Aggregation and Reduction Operations

def sum_elements(tensor):
    """Returns the sum of all elements in the tensor."""
    return tensor.sum()


def mean_elements(tensor):
    """Returns the mean of all elements in the tensor."""
    return tensor.float().mean()


def max_element(tensor):
    """Returns the maximum value in the tensor."""
    return tensor.max()


def min_element(tensor):
    """Returns the minimum value in the tensor."""
    return tensor.min()


def count_occurrences(tensor, value):
    """Counts the number of occurrences of a specific value in the tensor."""
    return torch.sum(tensor == value)


def dot_product(a, b):
    """Returns the dot product of two tensors."""
    return torch.dot(a, b)


# Tensor Transformation Operations

def transpose_tensor(tensor):
    """Transposes the tensor."""
    return tensor.T


def reshape_tensor(tensor, shape):
    """Reshapes the tensor to a new shape."""
    return tensor.view(shape)


def flatten_tensor(tensor):
    """Flattens the tensor to a 1D tensor."""
    return tensor.flatten()


# Statistical Functions

def variance(tensor):
    """Returns the variance of the tensor elements."""
    return tensor.var()


def standard_deviation(tensor):
    """Returns the standard deviation of the tensor elements."""
    return tensor.std()


def median(tensor):
    """Returns the median value of the tensor."""
    return tensor.median()


def covariance(a, b):
    """Returns the covariance between two tensors."""
    mean_a = a.mean()
    mean_b = b.mean()
    return ((a - mean_a) * (b - mean_b)).mean()


# Logical Functions

def logical_and(tensor1, tensor2):
    """Returns element-wise logical AND between two tensors."""
    return tensor1 & tensor2


def logical_or(tensor1, tensor2):
    """Returns element-wise logical OR between two tensors."""
    return tensor1 | tensor2


def logical_not(tensor):
    """Returns element-wise logical NOT of the tensor."""
    return ~tensor


def logical_xor(tensor1, tensor2):
    """Returns element-wise logical XOR between two tensors."""
    return tensor1 ^ tensor2


# Miscellaneous Functions

def norm(tensor, p=2):
    """Returns the p-norm of the tensor."""
    return tensor.norm(p)


def reshape_tensor(tensor, new_shape):
    """Reshapes the tensor to the new shape."""
    return tensor.view(new_shape)


def clip_tensor(tensor, min_value, max_value):
    """Clips the values of the tensor to the specified min and max."""
    return tensor.clamp(min_value, max_value)


# Example usage:

if __name__ == "__main__":
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    print("Addition: ", add_tensors(a, b))
    print("Subtraction: ", subtract_tensors(a, b))
    print("Dot Product: ", dot_product(a, b))
    print("Sum of elements in a: ", sum_elements(a))
    print("Mean of elements in a: ", mean_elements(a))
    print("Maximum in b: ", max_element(b))
    print("Count of occurrences of 2 in a: ", count_occurrences(a, 2))
    print("Variance of a: ", variance(a))
    print("Standard Deviation of a: ", standard_deviation(a))
    print("Median of b: ", median(b))
    print("Covariance of a and b: ", covariance(a, b))
    print("Logical AND: ", logical_and(a > 1, b < 6))
    print("Logical NOT: ", logical_not(a > 1))