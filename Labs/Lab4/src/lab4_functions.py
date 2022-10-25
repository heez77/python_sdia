import numpy as np

class K_nearest_neighbors():
    """K nearest neighbors implementation.
    """
    def __init__(self, train:np.ndarray, test:np.ndarray, K:int):
        """
        Args:
            train (np.ndarray): The training dataset
            test (np.ndarray): The test dataset
            K (int): K value in the nearest neighbors implementation, must be strictly positive.
        """
        self.x_train = train[:,1:]
        self.y_train = train[:,0]
        self.N_train = train.shape[0]
        self.x_test = test[:,1:]
        self.y_test = test[:,0]
        self.N_test = test.shape[0]
        assert K>0, "K must be strictly positive"
        self.K = K

    def classifier(self, x_new:np.ndarray)->float:
        """Compute the classification using K nearest neighbors algorithm method.

        Args:
            x_new (np.ndarray): The input to be predicted.

        Returns:
            float: 1. if x_new is in class 1 else 2.
        """
        distances = []
        counter1 = 0
        counter2 = 0
        for i in range(self.N_train):
            distances.append((np.linalg.norm(x_new-self.x_train[i]),self.y_train[i]))
        dtype = [('distance', float), ('target', float)]
        distances=np.array(distances,dtype=dtype)
        distances=np.sort(distances,order='distance')  # plus petites en premier
        for j in range(0,self.K):
            if distances[j][1]==1.:
                counter1+=1
            if distances[j][1]==2.:
                counter2+=1
            if counter1+counter2==self.K:
                break
        if counter1>counter2:
            return 1.
        else:
            return 2.

    def predict(self, on='test')->np.ndarray:
        """Compute the prediction for the test dataset.

        Returns:
            np.ndarray: An array in dimension 1 with all the predictions.
        """
        if on=='test':
            N = self.N_test
            x = self.x_test
        else:
            N= self.N_train
            x = self.x_train
        predictions = np.zeros(N, dtype=float)
        for i in range(N):
            predictions[i] = self.classifier(x[i])
        return predictions

    def error_rate(self,on='test')->float:
        """Compute the error rate for the test dataset.

        Returns:
            float: Error rate value, a float between 0 and 1.
        """
        predictions = self.predict(on=on)
        if on=='test':
            error_rate = 1-np.count_nonzero(predictions==self.y_test)/self.N_test
        else:
            error_rate = 1-np.count_nonzero(predictions==self.y_train)/self.N_train
        return error_rate
