import numpy as np
import scipy
import matplotlib.pyplot as plt

from tut3 import backSub3, forwardSub3


def lu_decomposition_with_partial_pivoting(A):
    A = A.copy()
    n = A.shape[0]

    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for i in range(n):

        pivot_row = i + np.argmax(np.abs(U[i:, i]))

        U[[i, pivot_row]] = U[[pivot_row, i]]

        if i > 0:
            L[[i, pivot_row], :i] = L[[pivot_row, i], :i]

        P[[i, pivot_row]] = P[[pivot_row, i]]

        for row in range(i + 1, n):
            ratio = U[row, i] / U[i, i]
            L[row, i] = ratio
            U[row, :] -= ratio * U[i, :]

    L += np.eye(n)

    return P, L, U


def solve_system(A, b, k=1):
    P, L, U = lu_decomposition_with_partial_pivoting(A)
    x = b
    for _ in range(k):
        x = forwardSub3(L, P @ x)
        x = backSub3(U, x)
    return x

def gauss_seidel(A, b, x0, max_iter=10, tol=1e-6):
    # Set up anything you need
    converged = False
    results = []
    P = np.zeros((A.shape[0], A.shape[1]))
    Q = np.zeros((A.shape[0], A.shape[1]))

    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if i >= j:
                P[i][j] = A[i][j]
            else:
                Q[i][j] = -A[i][j]

    x = x0.copy()
    for k in range(max_iter):
        for i in range(x.shape[0]):
            sum1 = sum(Q[i, j] * x[j] for j in range(i+1, x.shape[0]))
            sum2 = sum(P[i, j] * x[j] for j in range(i))
            x[i] = (b[i] + sum1 - sum2) / P[i, i]
        results.append([k+1, *x])

    return np.array(results)

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    B = np.array([
        [10.0, -4.0, -3.9, -5.9, 0.5],
        [1.0, -0.4, 2.4, -2.0, 3.7],
        [5.5, -2.2, 2.4, -2.0, 3.7],
        [9.9, -6.9, 5.0, 0.7, 6.2],
        [-7.5, 3.5, 5.9, 8.5, -6.2]
    ])
    C = scipy.linalg.orth(np.random.rand(5, 5))
    x_true = np.ones(5)
    relative_error_inf_norm_array = np.array([])
    print(np.linalg.cond(B))
    print(np.linalg.cond(C))
    # for k in range(200):
    #     b = np.linalg.matrix_power(B, k) @ x_true
    #     x_approx = solve_system(B, b, k)
    #     absolute_error = abs(x_approx - x_true)
    #     relative_error_inf_norm = np.max(absolute_error) / np.max(x_true)
    #     relative_error_inf_norm_array = np.append(relative_error_inf_norm_array, relative_error_inf_norm)
    #
    # fig, ax = plt.subplots()
    # ax.set_title('Relative Error in Infinite Norm vs k exponent')
    # ax.set_ylabel('Relative Error in Infinite Norm')
    # ax.set_xlabel('k')
    # ax.plot(np.arange(0, 200), relative_error_inf_norm_array)
    # plt.show()
