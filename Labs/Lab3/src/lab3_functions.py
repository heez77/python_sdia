# your code
from sympy import symbols, Eq, solve
import numpy as np


def dot(v: np.array, w: np.array) -> int:
    """Returns the scalar product of v and w (dimension 2)

    Args:
        v (array): array
        w (array): array

    Returns:
        int: scalar product of v and w
    """
    return v[0] * w[0] + v[1] * w[1]


def wedge(v: np.array, w: np.array) -> int:
    """Returns v[0]*w[1] - v[1]*w[0]

    Args:
        v (array): array
        w (array): array

    Returns:
        int: v[0]*w[1] - v[1]*w[0]
    """
    return v[0] * w[1] - v[1] * w[0]


def is_between(a: np.array, b: np.array, c: np.array) -> bool:
    """Returns True if b is between a and c otherwise returns False

    Args:
        a (array): lower bound
        b (array): point of interest
        c (array): upper bound

    Returns:
        bool: True if b is is between a and c otherwise False
    """
    v = a - b
    w = b - c
    return abs(wedge(v, w)) <= 10e-6 and dot(v, w) > 0


def brownian_motion(
    niter: int, x: np.array, step: int, rng: np.random._generator.Generator
) -> np.array:
    """The Brownian motion is a random walk with independent, identically distributed Gaussian increments, appearing for instance in thermodynamics and statistical mechanics (to model the evolution of a large particle in a medium composed of a large number of small particles, ...). It is also connected to the diffusion process (Einstein).
    generating a Brownian motion on the closed unit ball $\mathcal{B}(\mathbf{0}, 1) = \{ \mathbf{x} \mid \Vert \mathbf{x} \Vert  \leq 1\}$

    Args:
        niter (int): Maximum of iterations
        x (array): Starting point in the ball
        step (int): Step-size
        rng (numpy.random._generator.Generator): Random generator

    Returns:
        array: Trajectory of the random walk
    """
    n = 0
    W = x
    W_list = [W]
    assert step > 0, "step must be strictly positive"
    while n < niter and (W.T @ W < 1):
        n += 1
        W = W + np.sqrt(step) * np.random.multivariate_normal(
            [0 for _ in range(x.shape[0])], np.identity(x.shape[0])
        )
        if W.T @ W == 1:
            W_list.append(W)
            break
        elif W.T @ W < 1:
            W_list.append(W)
        else:
            x, y = symbols("x y")

            eq1 = Eq(x**2 + y**2, 1)
            a = (W[1] - W_list[-1][1]) / (W[0] - W_list[-1][0])
            b = W[1] - a * W[0]
            eq2 = Eq(a * x + b, y)
            sol_dict = solve((eq1, eq2), (x, y))
            sol_one = np.array(sol_dict[0])
            sol_two = np.array(sol_dict[1])
            if is_between(W_list[-1], sol_one, W):
                W_list.append(sol_one)
            else:
                W_list.append(sol_two)
    return np.stack(W_list, axis=0)


def ideal_lowpass_filter(x: np.array, fc: tuple) -> np.array:
    """Returns a low pass filter on a 2D array

    Args:
        x (array): The 2D array to be filtered
        fc ((float, float)): Cutoff frequencies

    Returns:
        array: The 2D array after the low pass filter
    """
    dft_x = np.fft.fft2(x)
    dft_x = np.fft.fftshift(dft_x)
    M, N = dft_x.shape
    dft_x_low_pass = dft_x[
        M // 2 - fc[0] // 2 : M // 2 + fc[0] // 2,
        N // 2 - fc[1] // 2 : N // 2 + fc[1] // 2,
    ]
    return np.fft.ifft2(dft_x_low_pass)


def gaussian_filter_2d(x: np.array, h: np.array) -> np.array:
    """_summary_

    Args:
        x (array): The 2D array to be filtered
        h (array): 2D Gaussian kernel obtained from the outer product of two gaussian windows

    Returns:
        array: The 2D array after the gaussian filter
    """
    M1 = x.shape[0]
    N1 = x.shape[1]
    M2 = h.shape[0]
    N2 = h.shape[1]
    M = M1 + M2 - 1
    N = N1 + N2 - 1

    P1_x = np.pad(x, ((0, M - M1), (0, N - N1)), "constant", constant_values=(0))
    P2_h = np.pad(h, ((0, M - M2), (0, N - N2)), "constant", constant_values=(0))
    dft_P1_x = np.fft.rfft2(P1_x)
    dft_P2_h = np.fft.rfft2(P2_h)
    hadamard = np.multiply(dft_P1_x, dft_P2_h)
    inv = np.fft.irfft2(hadamard)
    return inv[:M1, :N1]
