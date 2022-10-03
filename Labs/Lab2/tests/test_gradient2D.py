import unittest
from src import lab2_functions
import numpy as np

class Test_TestGradient2D(unittest.TestCase):
    def test_gradient2D_square(self):
        rng = np.random.default_rng(84548)
        N = rng.integers(low=2, high=20, size = 1)[0]
        X1 = rng.random((N,N))
        XDh1, DvX1 = lab2_functions.gradient2D(X1)
        self.assertCountEqual([XDh1.shape[0], XDh1.shape[1], DvX1.shape[0], DvX1.shape[1]], [N, N, N, N])

    def test_gradient2D_non_square(self):
        rng = np.random.default_rng(84548)
        M,N = rng.integers(low=2, high=20, size = 2)
        X2 = rng.random((M,N))
        XDh2, DvX2 = lab2_functions.gradient2D(X2)
        self.assertCountEqual([XDh2.shape[0], XDh2.shape[1], DvX2.shape[0], DvX2.shape[1]], [M, N, M, N])
