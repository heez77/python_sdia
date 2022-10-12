import unittest
from src import lab3_functions
import numpy as np
from scipy.signal import windows

class Test_TestIdealLowpassFilter(unittest.TestCase):
    def test_ideal_lowpass_filter_type(self):
        rng = np.random.default_rng(84548)
        M1,N1, M2, N2 = rng.integers(low=2, high=1024, size = 4)
        x = rng.integers(low=0, high=255, size=(M1,N1))

        w_x = windows.gaussian(N2,10).reshape((N2,1))
        w_y = windows.gaussian(M2,10).reshape((M2,1))
        h = w_y @ w_x.T
        self.assertIsInstance(
            lab3_functions.gaussian_filter_2d(x, h), np.ndarray
        )
    def test_ideal_lowpass_filter_shape_output(self):
        rng = np.random.default_rng(84548)
        M1,N1, M2, N2 = rng.integers(low=2, high=1024, size = 4)
        x = rng.integers(low=0, high=255, size=(M1,N1))

        w_x = windows.gaussian(N2,10).reshape((N2,1))
        w_y = windows.gaussian(M2,10).reshape((M2,1))
        h = w_y @ w_x.T
        self.assertEqual(
            lab3_functions.gaussian_filter_2d(x, h).shape, x.shape
        )