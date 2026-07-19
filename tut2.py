import numpy as np


def myRowEchelon3(A):
    """Takes a 3x3 matrix and uses Gaussian
    elimination to find its row echelon form"""
    A[1] -= A[1][0]/A[0][0] * A[0]
    A[2] -= A[2][0]/A[0][0] * A[0]
    A[2] -= A[2][1]/A[1][1] * A[1]
    return A


if __name__ == "__main__":
    A = np.array([[2., -1., 1.], [-2., 3., -1.], [4., -15., 7.]], dtype=float)
    np.random.seed(seed=211)
    B = np.random.rand(3, 3)
    C = np.array([[3,3,1], [7,-2,1], [1,4,1]], dtype=float)
    print(myRowEchelon3(A))
    print(myRowEchelon3(B))


`