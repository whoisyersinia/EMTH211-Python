import numpy as np


def lu_decomposition_with_partial_pivoting(A):
    """Takes LU decomposition problem with partial pivoting."""
    row_reduction = {}
    P = np.zeros_like(A)
    A = A.copy()
    remaining_rows = A.copy()
    row_pivots = []
    for i in range(A.shape[0]):
        row_pivot_num = np.argmax(np.abs(remaining_rows[:, i]))
        abs_max_pivot = remaining_rows[:, i][row_pivot_num]
        remaining_rows[row_pivot_num, :] = np.zeros_like(remaining_rows[row_pivot_num, :])
        P[i, row_pivot_num] = 1
        row_pivots.append(row_pivot_num)
        for row in range(A.shape[0]):
            if row not in row_pivots:
                ratio = A[row, i] / abs_max_pivot
                if ratio != 0:
                    for col in range(A.shape[1]):
                        A[row, col] -= ratio * A[row_pivot_num, col]
                        row_reduction[row, row_pivot_num] = row

    U = P @ A
    return U, P


if __name__ == "__main__":
    # A = np.array([
    #     [10.0, -4.0, -3.9, -5.9, 0.5],
    #     [1.0, -0.4, 2.4, -2.0, 3.7],
    #     [5.5, -2.2, 2.4, -2.0, 3.7],
    #     [9.9, -6.9, 5.0, 0.7, 6.2],
    #     [-7.5, 3.5, 5.9, 8.5, -6.2]
    # ])
    # print(lu_decomposition_with_partial_pivoting(A))
    A = np.array([[0, 2, -2, 2], [2, -1, 1, 0], [-2, 2, 0, -3], [-3, 3, 0, 0]])
    print(lu_decomposition_with_partial_pivoting(A))
