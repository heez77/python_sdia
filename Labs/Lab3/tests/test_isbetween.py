import unittest
from xmlrpc.client import Boolean
from src import lab3_functions
import numpy as np


class Test_TestIsBetween(unittest.TestCase):
    def test_isbetween_type(self):
        rng = np.random.default_rng(84548)
        N = 2
        a = rng.random((N,))
        b = rng.random((N,))
        c = rng.random((N,))
        self.assertIsInstance(lab3_functions.is_between(a, b, c), bool)
    
    def test_isbetween_true(self):
        a = np.array([0,0])
        b = np.array([0.5,0.5])
        c =  np.array([1,1])
        self.assertEqual(lab3_functions.is_between(a, b, c), True)
    
    def test_isbetween_false(self):
        a = np.array([0,0])
        b = np.array([0.5,2])
        c =  np.array([1,1])
        self.assertEqual(lab3_functions.is_between(a, b, c), False)


