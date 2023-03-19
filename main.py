import numpy as np

def densityp(Pl:np.matrix, Pr:np.matrix):
    return np.trace(Pl @ Pr) == 1

def phi(Pl:np.matrix, M:np.matrix, Pr:np.matrix):
    return np.trace(Pl@M@Pr@M.getH())

def semidefinitify(m:np.matrix):
    """take m and appling to its conjugate transpose, hence returning a semidefinte matrix

    Conditions for semidefinity: z* M z is real positive for any nonzero complex vector z

    Small proof for why this works:
    z* M M* z = (M* z)* (M* z); and so the inner product between them will cause negative
      by negative, imag to imag, making it real and positive

    Question for ted: wouldn't this make all M real? Isn't that a problem?
    """
    return m @ m.getH()



semidefinitify(np.matrix(np.random.rand(5,5) + 1j * np.random.rand(5,5)))

np.linalg.eigvals(semidefinitify(np.matrix([[1+8j, 2-8j],
                                            [0, 1]])))

semidefinitify(np.matrix([[1+8j, 2-8j],
                          [0, 1]]))

densityp(np.matrix([[1+2j, 3+3j],
                    [0, 1]]),
         np.matrix([[1-5j, 2+0j],
                    [1, 1]]))

type(np.matrix([[1-5j, 2+0j],
                [1, 1]]))


