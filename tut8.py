import numpy as np
from numpy import linalg as la
np.set_printoptions(precision=4)

A = np.array([[1., 2.], [2., -5.]])

# Find the eigenvalues and eigenvectors of A using a numpy function
# Check your output to ensure that the following print statement is correct!
eigvals, eigvecs = np.linalg.eig(A)
for i in range(len(eigvals)):
    print(f"An eigenvalue of A is {eigvals[i]:.4f} with corresponding eigenvector\n{eigvecs[:,i]}\n")

# Construct appropriate matrices D and P to diagonalise A.
D = np.diag(eigvals)
P = eigvecs
print(f"A can be diagonalised with the diagonal matrix D =\n{D}\n and invertible matrix P =\n{P}\n")

k = 5
# Raise A to the power of k using your above answers.
# np.linalg.matrix_power is forbidden in this question.
# We will forgive the use of np.linalg.inv this one time!
Ak = P @ np.diag(eigvals**k) @ np.linalg.inv(P)
print(f"A^{k} is \n{Ak}\n")

# Find the dominant eigenvalue-eigenvector pair from your above eigvals and eigvecs variables.
L1 = np.max(np.abs(eigvals))
v1 = P[:, np.argmax(np.abs(eigvals))]
print(f"The dominant eigenvalue is {L1:.4f} with corresponding eigenvector\n{v1}\n")

