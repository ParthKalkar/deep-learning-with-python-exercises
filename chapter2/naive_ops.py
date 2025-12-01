import numpy as np

# Task: Naive Vector Operations
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

# Example usage
x = np.array([[1, -2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
print("naive_relu(x):", naive_relu(x))
print("naive_add(x, y):", naive_add(x, y))