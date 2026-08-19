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
    A = np.array([[2, 0.5, -0.5], [0, 4, - 2], [4, 0, - 4]])
    b = np.array([4., 4., 0.])
    x0 = np.array([1., 0., 0.])
    print(gauss_seidel(A, b, x0))
