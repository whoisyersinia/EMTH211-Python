import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd


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


def good_matrix(n, d):
    rng = np.random.default_rng(seed=211)
    B = rng.uniform(size=(n, n))
    return B.T @ B + d * n * np.eye(n)


def sor(A, b, x0, w, max_iter=1000, tol=1e-8):
    iter = 0

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
        x_prev = x.copy()
        for i in range(x.shape[0]):
            sum1 = sum(Q[i, j] * x[j] for j in range(i + 1, x.shape[0]))
            sum2 = sum(P[i, j] * x[j] for j in range(i))
            x[i] = (1-w)*x[i] + (w*(b[i] + sum1 - sum2)) / (P[i, i])

        iter += 1

        err = np.linalg.norm(x - x_prev, np.inf) / np.linalg.norm(x, np.inf)
        if err < tol:
            break

    return iter, x


if __name__ == "__main__":

    # n = 20
    # optimal_w = []
    # ratios = []
    # d_values = np.linspace(0.01, 2, 50)
    #
    # max_iter = 1000
    # for d in d_values:
    #     A = good_matrix(n, d)
    #     b = np.ones(n)
    #     x0 = np.ones(n)
    #     iter_array = []
    #     w_values = np.linspace(0.01, 1.99, 300)
    #     for w in w_values:
    #         iter_array.append(sor(A, b, x0, w, max_iter=max_iter)[0])
    #
    #     iter_array = np.array(iter_array)
    #     gs_iters = sor(A, b, x0,1)[0]
    #     best_idx = np.argmin(iter_array)
    #     optimal_w.append(w_values[best_idx])
    #     ratio_sor_to_gs = iter_array[best_idx] / gs_iters
    #     ratios.append(ratio_sor_to_gs)
    #     print(f"{d}: Ratio = {ratio_sor_to_gs:.4f}, Best Omega = {w_values[best_idx]:.4f}")
    #
    # plt.plot(d_values, optimal_w, color='r')
    # plt.xlabel('D values')
    # plt.ylabel("Optimal ω")
    # plt.show()
    # plt.plot(d_values, ratios,  color='b')
    # plt.xlabel('D values')
    # plt.ylabel("Ratio between SOR iterations / Gauss-seidel")
    # plt.show()

    A = good_matrix(20, 0.01)
    b = np.ones(20)
    x0 = np.ones(20)
    iter_array = []

    w_values = np.linspace(0.01, 1.99, 300)
    for w in w_values:
        iter_array.append((sor(A, b, x0, w))[0])

    iter_array = np.array(iter_array)
    gs_iters = sor(A, b, x0, 1)[0]
    better_mask = iter_array < gs_iters
    print(w_values[better_mask])
    print(w_values[better_mask].min(), w_values[better_mask].max())  # range: 0.295 to 0.990

    best_idx = np.argmin(iter_array)
    print(w_values[best_idx], iter_array[best_idx])

    for w, it in zip(w_values, iter_array):
        print(f"{w:.4f}: {it}")

    plt.axhline(y=sor(A, b, x0, 1)[0], color='r', linestyle='--', linewidth=2)
    plt.plot(w_values, iter_array)
    plt.ylabel('Number of iterations')
    plt.xlabel('Omega Values')
    plt.legend(['Gauss-seidel', 'SOR'])
    plt.show()

    # np.set_printoptions(precision=4, suppress=True)
    # B = np.array([
    #     [10.0, -4.0, -3.9, -5.9, 0.5],
    #     [1.0, -0.4, 2.4, -2.0, 3.7],
    #     [5.5, -2.2, 2.4, -2.0, 3.7],
    #     [9.9, -6.9, 5.0, 0.7, 6.2],
    #     [-7.5, 3.5, 5.9, 8.5, -6.2]
    # ])
    # B = scipy.linalg.orth(np.random.rand(5, 5))
    # x_true = np.ones(5)
    # relative_error_inf_norm_array = np.array([])
    # for k in range(200):
    #     b = np.linalg.matrix_power(B, k) @ x_true
    #     x_approx = solve_system(B, b, k)
    #     absolute_error = abs(x_approx - x_true)
    #     relative_error_inf_norm = np.max(absolute_error) / np.max(x_true)
    #     print(k, relative_error_inf_norm)
    #     relative_error_inf_norm_array = np.append(relative_error_inf_norm_array, relative_error_inf_norm)
    #
    # fig, ax = plt.subplots()
    # ax.set_title('Relative Error in Infinite Norm vs k exponent')
    # ax.set_ylabel('Relative Error in Infinite Norm')
    # ax.set_xlabel('k')
    # ax.plot(np.arange(0, 200), relative_error_inf_norm_array)
    # plt.show()



