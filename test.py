import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment

def generate_A_P(n):

    """Generate a random matrix associated wioth the P component of the LOP.

    A random permutation determines pairwise differences along a chain; the
    remaining differences are filled in to satisfy d_ij + d_jk = d_ik.
    Matrix entries are then chosen consistently with those differences.

    Args:
        n: Number of rows and columns.

    Returns:
        An n-by-n NumPy array A such that test_A_P(A) is True.
    """

    A = np.zeros((n, n))
    D = np.zeros((n, n))

    # Generate a random permutation
    sigma = np.random.permutation(n)

    # For i=1,2,...,n-1 do randomly these steps:
    for i in range(n-1):
        step = np.random.choice([1, 2, 3])
        
        if step == 1:
            # Assign a_{sigma(i)sigma(i+1)} and a_{sigma(i+1)sigma(i)} a random value respectively.
            a1 = np.random.uniform(-1,1)
            a2 = np.random.uniform(-1,1)

            # Set d_{sigma(i)sigma(i+1)} and d_{sigma(i+1)sigma(i)}.
            d1 = a1 - a2
            d2 = a2 - a1
        
        if step == 2:
            # Assign a_{sigma(i)sigma(i+1)} and d_{sigma(i)sigma(i+1)} a random value respectively.
            a1 = np.random.uniform(-1,1)
            d1 = np.random.uniform(-1,1)

            # Set a_{sigma(i+1)sigma(i)} and d_{sigma(i+1)sigma(i)}.
            d2 = -d1
            a2 = d2 + a1

        if step == 3:
            # Assign d_{sigma(i)sigma(i+1)} and a_{sigma(i+1)sigma(i)} a random value respectively.
            d1 = np.random.uniform(-1,1)
            a2 = np.random.uniform(-1,1)

            # Set d_{sigma(i+1)sigma(i)} and a_{sigma(i)sigma(i+1)}.
            d2 = -d1
            a1 = d1 + a2

        A[sigma[i], sigma[i+1]] = a1
        A[sigma[i+1], sigma[i]] = a2
        D[sigma[i], sigma[i+1]] = d1
        D[sigma[i+1], sigma[i]] = d2

    # Set the rest of the values of D, so that d_ij + d_jk = d_ik and d_ij = -d_ji for all i,j,k.
    while np.count_nonzero(D) < n*(n-1):
        for i,k in zip(*np.where(D == 0)):
            for j in range(n):
                if D[i, j] != 0 and D[j, k] != 0:
                    D[i, k] = D[i, j] + D[j, k]
                    D[k, i] = -D[i, k]
                    break
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                state = D[i, j] + D[j, k] - D[i, k]
                if not np.isclose(state, 0):
                    print(f"Warning: d_{i}{j} + d_{j}{k} != d_{i}{k}: {D[i, j]} + {D[j, k]} != {D[i, k]}")

    # Until A in completely defined, choose a random couple of indices i and j for which a_ij is not defined.
    for i,j in zip(*np.where(A == 0)):
        # Set a following uniform distribution U(-1,1) and a_ji = d_ji + a_ij.
        a_ij = np.random.uniform(-1, 1)
        a_ji = D[j, i] + a_ij
        A[i, j] = a_ij
        A[j, i] = a_ji

    return A


def test_A_P(A):
    D = A - A.T
    test = D[:, :, None] + D[None, :, :] - D[:, None, :]
    return np.allclose(test, 0)