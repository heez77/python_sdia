import unittest
from src.lab5_functions import markov
import numpy as np


class TestMarkov(unittest.TestCase):
    def test_markov_trajectory_type(self):
        rng = np.random.default_rng(50)
        rho = np.array([0.05, 0.2, 0.5, 0.2, 0.05])
        A = np.array(
            [
                [0.1, 0.2, 0.6, 0.05, 0.05],
                [0.2, 0.6, 0.05, 0.05, 0.1],
                [0.6, 0.05, 0.05, 0.1, 0.2],
                [0.05, 0.1, 0.2, 0.6, 0.05],
                [0.1, 0.2, 0.6, 0.05, 0.05],
            ]
        )
        nmax = rng.integers(low=1, high=20, size=1)[0]
        self.assertIsInstance(markov(rho, A, nmax, rng), np.ndarray)

    def test_markov_trajectory_size(self):
        rng = np.random.default_rng(50)
        rho = np.array([0.05, 0.2, 0.5, 0.2, 0.05])
        A = np.array(
            [
                [0.1, 0.2, 0.6, 0.05, 0.05],
                [0.2, 0.6, 0.05, 0.05, 0.1],
                [0.6, 0.05, 0.05, 0.1, 0.2],
                [0.05, 0.1, 0.2, 0.6, 0.05],
                [0.1, 0.2, 0.6, 0.05, 0.05],
            ]
        )
        nmax = rng.integers(low=1, high=20, size=1)[0]
        self.assertEqual(len(markov(rho, A, nmax, rng)), nmax + 1)
