import numpy as np


def myLU3(A):
    """ Takes a 3x3 numpy array and computes its
    LU decomposition without partial pivoting
    and assuming that no row swaps are required . """
    row_reduction = {}
    copy_A = np.copy(A)
    for row_piv in range(copy_A.shape[0]):
        pivot = copy_A[row_piv, row_piv]
        for row in range(row_piv + 1, copy_A.shape[0]):
            ratio = copy_A[row, row_piv] / pivot
            for col in range(copy_A.shape[1]):
                copy_A[row, col] -= ratio * copy_A[row_piv, col]
                row_reduction[(row, row_piv)] = ratio
    U = copy_A
    L = np.eye(A.shape[0])

    for pos, val in row_reduction.items():
        row = pos[0]
        col = pos[1]
        L[row, col] = val

    return L, U


def forwardSub3(L, b):
    """ Takes a 3x3 numpy array and a 1d numpy array
    and performs forward substitution """
    x = np.zeros(L.shape[0])

    for row in range(L.shape[0]):
        total = 0
        for col in range(L.shape[1]):
            total += L[row, col] * x[col]
        ratio = 1/L[row, row]
        x[row] = ratio*(b[row] - total)

    return x


def backSub3(U, b):
    """ Takes a 3x3 numpy array and a 1d numpy array
    and performs back substitution """
    x = np.zeros(U.shape[0])

    for row in range(U.shape[0]-1, -1, -1):
        total = 0
        for col in range(U.shape[1]-1, -1, -1):
            total += U[row, col] * x[col]

        x[row] = 1 / (U[row, row]) * (b[row] - total)

    return x


if __name__ == "__main__":
    # A = np.array([[-3, -3, -1], [9, 7, 0], [-6, 0, 9]], dtype=float)
    # L, U = myLU3(A)
    # print(f"L is\n{L}\nU is\n{U}\nL @ U == A: {np.allclose(A, L @ U)}")

    L = np.array([[1,0,0], [-2,1,0], [-3,0,1]], dtype=float)
    U = np.array([[-3,0,1], [0,1,3], [0,0,-1]], dtype=float)
    b = np.array([2,-6,-5], dtype=float)
    y = forwardSub3(L, b)
    print(y)
    x = backSub3(U, y)
    print(x)
    A = L @ U
    print(f"x is {x}. Does Ax = b? {np.allclose(A @ x, b)}")
