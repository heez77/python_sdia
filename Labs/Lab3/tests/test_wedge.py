import unittest
from src import lab3_functions
import numpy as np


class Test_TestWedge(unittest.TestCase):
    def test_wedge_type(self):
        rng = np.random.default_rng(84548)
        v = rng.random((2,))
        w = rng.random((2,))
        self.assertIsInstance(lab3_functions.wedge(v, w), float)
    def test_wedge_self(self):
        rng = np.random.default_rng(84548)
        v = rng.random((2,))
        c = rng.integers(low=1, high=100, size=1)[0]
        self.assertEqual(lab3_functions.wedge(c*v, v), 0)
