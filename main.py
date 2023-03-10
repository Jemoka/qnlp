import numpy as np


def densityp(Pl, Pr):
    return np.trace(Pl @ Pr) == 1

# def semidefinite()

np.linalg.eigvals(np.array([[1+2j, 2+3j],
                            [0, 1]]))

densityp(np.array([[1+2j, 3+3j],
                   [0, 1]]),
         np.array([[1-5j, 2+0j],
                   [1, 1]]))


