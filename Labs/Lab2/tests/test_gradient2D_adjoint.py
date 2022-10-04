import unittest
import numpy as np
from src import lab2_functions


class Test_TestGradient_2D_adjoint(unittest.TestCase):
    def test_gradient2D_adjoint_shape_square(self):
        rng = np.random.default_rng(84548)
        N = rng.integers(low=2, high=20, size = 1)[0]
        Yh1, Yv1 = rng.random((N,N)), rng.random((N,N))
        D_adjoint = lab2_functions.gradient2D_adjoint((Yh1, Yv1))

        self.assertCountEqual([D_adjoint.shape[0], D_adjoint.shape[1]], [Yh1.shape[0], Yh1.shape[1]])

    def test__gradient2D_adjoint_shape_non_square(self):
        rng = np.random.default_rng(84548)
        M,N = rng.integers(low=2, high=20, size = 2)
        Yh2, Yv2 = rng.random((M,N)), rng.random((M,N))
        D_adjoint = lab2_functions.gradient2D_adjoint((Yh2, Yv2))

        self.assertCountEqual([D_adjoint.shape[0], D_adjoint.shape[1]], [Yh2.shape[0], Yv2.shape[1]])

    def test_gradient2D_adjoint_equation(self):
        rng = np.random.default_rng(42)
        M,N = rng.integers(low=2, high=20, size = 2)
        X3, Yh3, Yv3 = rng.random((M,N)), rng.random((M,N)), rng.random((M,N))
        XDh3, DvX3 = lab2_functions.gradient2D(X3)
        D_adjointY3 = lab2_functions.gradient2D_adjoint((Yh3, Yv3))
        self.assertLessEqual(np.abs(np.trace(XDh3.T @ Yh3) + np.trace(DvX3.T @ Yv3)-np.trace(X3.T @ D_adjointY3)), 10E-6)
