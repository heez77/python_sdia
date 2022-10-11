import unittest
from src import lab3_functions
import numpy as np


class Test_TestBrownianMotion(unittest.TestCase):
    def test_brownian_motion_type(self):
        rng = np.random.default_rng(84548)
        niter = rng.integers(low=500, high=1000, size=1)[0]
        step = rng.integers(low=10e-6, high=10e-2, size=1)[0]
        x = rng.random((2,))
        self.assertIsInstance(
            lab3_functions.brownian_motion(niter, x, step, rng), np.array
        )
