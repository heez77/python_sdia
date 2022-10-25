import numpy as np


def markov(rho, A, nmax, rng):
    X = [rho]
    N = rho.shape[0]
    niter = 0

    assert A.shape[0] == N and A.shape[1] == N, "The transition matrix must be squared"
    assert (sum(rho) - 1.0) < 1e-3, "The components of rho must be summing to one"
    for _ in rho:
        assert _ >= 0, "rho must be a nonnegative vector"

    nb_points = 1000
    values = np.linspace(1, N, num=N, dtype=int)

    # draw_variable(values, rho, nb_points)

    while niter < nmax:
        niter += 1
        rho = A.T @ rho
        X.append(rho)
        # draw_variable(values, rho, nb_points)

    return X
