import numpy as np


def jacobi(A, b, x0, max_iter=10, tol=1e-6):
    # Set up anything you need
    converged = False
    results = []
    P = np.zeros((A.shape[0], A.shape[1]))
    Q = np.zeros((A.shape[0], A.shape[1]))
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if i == j:
                P[i][j] = A[i][j]
            else:
                Q[i][j] = -A[i][j]

    x_prev = x0
    x = np.zeros(b.shape[0])
    for k in range(max_iter):
        for i in range(x.shape[0]):
            elements_sum = 0
            for j in range(x.shape[0]):
                if i != j:
                    elements_sum += Q[i][j] * x_prev[j]
            x[i] = (1 / P[i][i]) * (elements_sum + b[i])
        err = np.linalg.norm(x - x_prev, np.inf)/np.linalg.norm(x, np.inf)
        if err < tol:
            converged = True
            break
        x_prev = x.copy()
        results.append([k + 1, *x])
    return np.array(results), converged


def gauss_seidel(A, b, x0, max_iter=1000, tol=1e-8):
    # Set up anything you need
    iter = 0

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
        x_prev = x.copy()

        for i in range(x.shape[0]):
            sum1 = sum(Q[i, j] * x[j] for j in range(i+1, x.shape[0]))
            sum2 = sum(P[i, j] * x[j] for j in range(i))
            x[i] = (b[i] + sum1 - sum2) / P[i, i]
        results.append([k+1, *x])

        iter += 1

        err = np.linalg.norm(x - x_prev, np.inf) / np.linalg.norm(x, np.inf)
        if err < tol:
            break

    return iter, x


def good_matrix(n, d):
    rng = np.random.default_rng(seed=211)
    B = rng.uniform(size=(n, n))
    return B.T @ B + d * n * np.eye(n)


if __name__ == "__main__":
    A = good_matrix(20, 0.5)
    b = np.ones(20)
    x0 = np.ones(20)
    print(gauss_seidel(A, b, x0))
