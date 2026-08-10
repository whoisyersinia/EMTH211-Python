import numpy as np
from tut3 import myLU3, forwardSub3, backSub3


def solve_system3(A, k, b):
    """Solves the system A**k @ x = for 3x3 A"""
    L, U = myLU3(A)
    last_sol = b
    for _ in range(k):
        last_sol = forwardSub3(L, last_sol)
        last_sol = backSub3(U, last_sol)
    return last_sol


if __name__ == "__main__":
    np.random.seed(seed=211)
    A = np.random.random((3, 3))
    b = np.ones(3)
    k = 5

    x = solve_system3(A, k, b)
    print(f"{x = }")
    print(np.allclose(np.linalg.matrix_power(A, k) @ x, b))
