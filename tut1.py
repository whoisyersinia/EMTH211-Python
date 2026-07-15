import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

A = np.array([[1, 1, 1],
              [1, 2, 3], [1, 3, 6]])
b = np.array([3,1,4])
x = la.solve(A, b)
print(x)
print(A @ x)