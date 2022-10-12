import unittest
from src import lab3_functions
import numpy as np


class Test_TestBrownianMotion(unittest.TestCase):
    def test_brownian_motion_type(self):
        rng = np.random.default_rng(84548)
        niter = rng.integers(low=500, high=1000, size=1)[0]
        step = (10e-2 - 10e-6)*rng.random() + 10e-6
        x = np.array([0.2,0.4])
        self.assertIsInstance(
            lab3_functions.brownian_motion(niter, x, step, rng), np.ndarray
        )
    def test_brownian_motion_in_ball(self):
        """ Test if all of the points are in the ball"""
        rng = np.random.default_rng(84548)
        niter = rng.integers(low=500, high=1000, size=1)[0]
        step = (10e-2 - 10e-6)*rng.random() + 10e-6
        x = np.array([0.2,0.4])
        list_points =  lab3_functions.brownian_motion(niter, x, step, rng)
        self.assertEqual(False not in (1-np.diagonal(list_points @ list_points.T) >= np.zeros(list_points.shape[0])), True)
        
    def test_brownian_motion_last_point_(self):
        """ Test if the last point intersect the ball"""
        rng = np.random.default_rng(84548)
        niter = rng.integers(low=500, high=1000, size=1)[0]
        step = (10e-2 - 10e-6)*rng.random() + 10e-6
        x = np.array([0.2,0.4])
        list_points =  lab3_functions.brownian_motion(niter, x, step, rng)
        self.assertLessEqual(np.abs(1-list_points[-1,:].T @ list_points[-1,:]), 10e-6)