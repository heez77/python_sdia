import unittest
from src import lab3_functions
import numpy as np


class Test_TestDot(unittest.TestCase):
    def test_dot_type(self):
        rng = np.random.default_rng(84548)
        v = rng.random((2,))
        w = rng.random((2,))
        self.assertIsInstance(lab3_functions.dot(v, w), int)
