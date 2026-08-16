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


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    # B = np.array([
    #     [10.0, -4.0, -3.9, -5.9, 0.5],
    #     [1.0, -0.4, 2.4, -2.0, 3.7],
    #     [5.5, -2.2, 2.4, -2.0, 3.7],
    #     [9.9, -6.9, 5.0, 0.7, 6.2],
    #     [-7.5, 3.5, 5.9, 8.5, -6.2]
    # ])
    B = scipy.linalg.orth(np.random.rand(5, 5))
    x_true = np.ones(5)
    relative_error_inf_norm_array = np.array([])
    for k in range(200):
        b = np.linalg.matrix_power(B, k) @ x_true
        x_approx = solve_system(B, b, k)
        absolute_error = abs(x_approx - x_true)
        relative_error_inf_norm = np.max(absolute_error) / np.max(x_true)
        print(k, relative_error_inf_norm)
        relative_error_inf_norm_array = np.append(relative_error_inf_norm_array, relative_error_inf_norm)

    fig, ax = plt.subplots()
    ax.set_title('Relative Error in Infinite Norm vs k exponent')
    ax.set_ylabel('Relative Error in Infinite Norm')
    ax.set_xlabel('k')
    ax.plot(np.arange(0, 200), relative_error_inf_norm_array)
    plt.show()
