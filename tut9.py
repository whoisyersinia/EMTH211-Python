import numpy as np

def power_method(A, x0, max_iter=100, tol=1e-6):
    """Finds an eigenvector associated with the dominant eigenvalue"""
    prev_x = x0
    x = prev_x
    for i in range(max_iter):
        y = A @ prev_x
        x = y/np.linalg.norm(y, np.inf)
        r = np.linalg.norm(x - prev_x, np.inf) / np.linalg.norm(x, np.inf)
        if r < tol:
            break
        prev_x = x
    return x

if __name__ == "__main__":
    n=3
    np.random.seed(seed=211)
    A = np.random.randn(n,n)
    x0 = np.ones(n)

    x = power_method(A, x0)
    Ax = A @ x
    R  = Ax.dot(x) / x.dot(x)
    print(f"{x = }")
    print(f"A @ x = {Ax}")
    print(f"lambda * x = {R * x}")
