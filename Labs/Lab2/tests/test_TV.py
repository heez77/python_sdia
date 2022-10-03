import unittest
import numpy as np
from src import lab2_functions

class Test_TestTV(unittest.TestCase):
    def test_TV_square(self):
        rng = np.random.default_rng(84548)
        N = rng.integers(low=2, high=20, size = 1)[0]
        X1 = rng.random((N,N))
        self.assertIsInstance(lab2_functions.tv(X1), float)

    def test_TV_non_square(self):
        rng = np.random.default_rng(84548)
        M,N = rng.integers(low=2, high=20, size = 2)
        X2 = rng.random((M,N))
        self.assertIsInstance(lab2_functions.tv(X2), float)

    def test_TV_constant(self):
        rng = np.random.default_rng(84548)
        M,N, C = rng.integers(low=2, high=20, size = 3)
        X3 = np.full((M,N),C)
        self.assertEqual(lab2_functions.tv(X3), 0)
