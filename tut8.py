import numpy as np
from numpy import linalg as la

np.set_printoptions(precision=4)
# A = np.array([[1., 2.], [2., -5.]])

A = np.array ([
[ 1., 3., 0. , 0., 0.],
[-3., -2., 0., 0. , 0.],
[ 0., 0., 2. , 3., 0.],
[ 0., 0., -3., -1. , 0.],
[ 0., 0., 0. , 0., 1.]
])


"""
B = np. array ([
[0.32 , 0.27 , 0.91 , 0. 81] ,
[0.29 , 0.74 , 0.34 , 0. 25] ,
[0.98 , 0.29 , 0.44 , 0. 12] ,
[0.45 , 0.27 , 0.66 , 0. 91]
])
C = np. diag ([ 0.1, -0.2,0.3, -0.99 , -0.5 ,0.4])
"""
# Find the eigenvalues and eigenvectors of A using a numpy function
# Check your output to ensure that the following print statement is correct !
eigvals, eigvecs = np.linalg.eig(A)

for i in range(len(eigvals)):
    print (f"An eigenvalue of A is {eigvals[i]:.4f} with corresponding eigenvector \n{eigvecs[: ,i]}\n")

# Construct appropriate matrices D and P to diagonalise A.
D = eigvals
P = eigvecs
print(f"A can be diagonalised with the diagonal matrix D =\n{D}\n and invertible matrix P =\n{P}\n")

k = 1
# Raise A to the power of k using your above answers .
# Avoid np. linalg . matrix_power !

Ak = P @ D**k @ P.T
print(f"A^{k} is \n{Ak}\n")

# Find the dominant eigenvalue - eigenvector pair from your above eigvals and eigvecs variables.
L1 = np.max(np.abs(D))
v1 = P[:, np.argmax(np.abs(D))]
print (f" The dominant eigenvalue is {L1:.4f} with corresponding eigenvector \n{v1}\n")