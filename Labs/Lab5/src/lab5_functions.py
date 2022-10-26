import numpy as np


def markov(rho, A, nmax, rng):
    """Computes a Markov chain based on a transition matrix and initial conditions

    Args:
        rho (np.ndarray): Law of the initial state (nonnegative vector of size N, summing to 1)
        A (np.ndarray): Transition matrix (squared)
        nmax (int): Number of time steps
        rng (np.random._generator.Generator): Random generator

    Returns:
        X: Trajectory of the chain
    """

    assert (
        abs(np.sum(rho) - 1.0) <= 1e-9
    ), "summing of rho composants must be equals to 1"
    assert False not in list(
        np.abs(np.sum(A, axis=1) - np.full(A.shape[0], 1.0))
        <= np.full(A.shape[0], 1e-9)
    ), "summing of each row of A must be equals to 1"
    assert A.shape[0] == N and A.shape[1] == N, "The transition matrix must be squared"
    for _ in rho:
        assert _ >= 0, "rho must be a nonnegative vector"

    X = [rho]
    N = rho.shape[0]
    niter = 0

    nb_points = 1000
    values = np.linspace(1, N, num=N, dtype=int)

    while niter < nmax:
        niter += 1
        rho = rho.T @ A
        X.append(rho)

    return X
