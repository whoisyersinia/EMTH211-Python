import numpy as np

def myGramSchmidt(basis):
    """
    Orthogonalises a given basis using the Gram-Schmidt process.
    Takes a list of 1D numpy arrays representing the basis.
    """
    ortho_basis = []
    for vec in basis:
        projection_sums = 0
        for i in range(len(ortho_basis)):
            projection_sums -= (np.dot(vec, ortho_basis[i])/np.dot(ortho_basis[i], ortho_basis[i])) * ortho_basis[i]
        ortho_basis.append(vec.copy() + projection_sums)
    return ortho_basis

if __name__ == "__main__":
    basis = [
        np.array([1., 1., -1., -1. ]),
        np.array([-1., -4., 4., 1.]),
        np.array([2., -3., 1., 0.])
    ]
    orth_basis = myGramSchmidt(basis)
    print(f"The new, orthogonal basis is\n{np.array(orth_basis)}\n")

