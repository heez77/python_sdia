import unittest
from src import lab3_functions
import numpy as np


class Test_TestIsBetween(unittest.TestCase):
    def test_isbetween_type(self):
        rng = np.random.default_rng(84548)
        N = rng.integers(low=2, high=20, size=1)[0]
        a = rng.random((N,))
        b = rng.random((N,))
        c = rng.random((N,))
        self.assertIsInstance(lab3_functions.is_between(a, b, c), bool)
