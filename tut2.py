import numpy as np
import numpy.linalg as la


def myRowEchelon3(A):
    """Takes a 3x3 matrix and uses Gaussian
    elimination to find its row echelon form"""
    A[1] -= A[1][0]/A[0][0] * A[0]
    A[2] -= A[2][0]/A[0][0] * A[0]
    A[2] -= A[2][1]/A[1][1] * A[1]
    return A


if __name__ == "__main__":
    A = np.array([[1, 1, 0], [0, -4, 1], [0,3,0]], dtype=float)
    x = np.identity(3)
    B = la.solve(A, x)
    print(B)
    print(A @ B)




