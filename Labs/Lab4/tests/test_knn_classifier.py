import unittest
from src.lab4_functions import K_nearest_neighbors
import numpy as np

class Test_Knn(unittest.TestCase):
    """
    We check if the classification for one entry returns a float.
    """
    def test_classifier_type(self):
        train = np.loadtxt('data/synth_train.txt')
        test = np.loadtxt('data/synth_test.txt')
        rng = np.random.default_rng(84548)
        K = rng.integers(low=1, high=20, size = 1)[0]
        i = rng.integers(low=1, high=test.shape[0], size = 1)[0]
        knn_classifier = K_nearest_neighbors(train, test, K)
        self.assertIsInstance(knn_classifier.classifier(test[i,1:]), float)

    def test_classifier_result(self):
        """
        We check if the classification for one entry returns 1. or 2. (referenced to class 1 or class 2).
        """
        train = np.loadtxt('data/synth_train.txt')
        test = np.loadtxt('data/synth_test.txt')
        rng = np.random.default_rng(84548)
        K = rng.integers(low=1, high=20, size = 1)[0]
        i = rng.integers(low=1, high=test.shape[0], size = 1)[0]
        knn_classifier = K_nearest_neighbors(train, test, K)
        prediction = knn_classifier.classifier(test[i, 1:])
        self.assertEqual((prediction==1.) | (prediction==2.), True)

    def test_prediction_type(self):
        """
        We check if the prediction on the full test dataset returns a numpy array.
        """
        train = np.loadtxt('data/synth_train.txt')
        test = np.loadtxt('data/synth_test.txt')
        rng = np.random.default_rng(84548)
        K = rng.integers(low=1, high=20, size = 1)[0]
        knn_classifier = K_nearest_neighbors(train, test, K)
        predictions = knn_classifier.predict()
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_prediction_size(self):
        """
        We check if the prediction on the full test dataset has the same size as the test dataset.
        """
        train = np.loadtxt('data/synth_train.txt')
        test = np.loadtxt('data/synth_test.txt')
        rng = np.random.default_rng(84548)
        K = rng.integers(low=1, high=20, size = 1)[0]
        knn_classifier = K_nearest_neighbors(train, test, K)
        predictions = knn_classifier.predict()
        self.assertEqual(predictions.shape[0], test.shape[0])
        
    def test_error_rate_type(self):
        """
        We check it the error rate is a float.
        """
        train = np.loadtxt('data/synth_train.txt')
        test = np.loadtxt('data/synth_test.txt')
        rng = np.random.default_rng(84548)
        K = rng.integers(low=1, high=20, size = 1)[0]
        knn_classifier = K_nearest_neighbors(train, test, K)
        error_rate = knn_classifier.error_rate()
        self.assertIsInstance(error_rate, float)

    def test_error_rate_is_between(self):
        """
        We check if the error rate is between 0. and 1. because it is a rate.
        """
        train = np.loadtxt('data/synth_train.txt')
        test = np.loadtxt('data/synth_test.txt')
        rng = np.random.default_rng(84548)
        K = rng.integers(low=1, high=20, size = 1)[0]
        knn_classifier = K_nearest_neighbors(train, test, K)
        error_rate = knn_classifier.error_rate()
        self.assertEqual(error_rate<=1. and error_rate>=0., True)

    def test_error_rate_on_train(self):
        """
        We check if the error rate is 0 on the training dataset with K=1 because for each x in x_train his closest neighbor is itself.
        """
        train = np.loadtxt('data/synth_train.txt')
        test = np.loadtxt('data/synth_test.txt')
        K = 1
        knn_classifier = K_nearest_neighbors(train, test, K)
        error_rate = knn_classifier.error_rate(on='train')
        self.assertEqual(error_rate==0., True)