import unittest
from src import lab3_functions
import numpy as np


class Test_TestIdealLowpassFilter(unittest.TestCase):
    def test_ideal_lowpass_filter_type(self):
        rng = np.random.default_rng(84548)
        M,N = rng.integers(low=2, high=20, size = 2)
        f = rng.integers(low=1, high=min(M,N), size = 1)[0]
        fc = np.array([f, f])
        x = rng.integers(low=0, high=255, size=(M,N))
        self.assertIsInstance(
            lab3_functions.ideal_lowpass_filter(x, fc), np.ndarray
        )
    def test_ideal_lowpass_filter_shape_output(self):
        rng = np.random.default_rng(84548)
        M,N = rng.integers(low=2, high=20, size = 2)
        f = rng.integers(low=1, high=min(M,N), size = 1)[0]
        fc = np.array([f, f])
        x = rng.integers(low=0, high=255, size=(M,N))
        self.assertEqual(
            lab3_functions.ideal_lowpass_filter(x, fc).shape, (f,f)
        )