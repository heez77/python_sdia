import numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.intc


cdef float classifier(train, test, K, x_new):
    cdef distances = []
    cdef int counter1 = 0
    cdef int counter2 = 0
    cdef x_train = train[:,1:]
    cdef y_train = train[:,0]
    cdef N_train = train.shape[0]
    for i in range(N_train):
        distances.append((np.linalg.norm(x_new-x_train[i]),y_train[i]))
    dtype = [('distance', float), ('target', float)]
    distances = np.array(distances,dtype=dtype)
    distances = np.sort(distances,order='distance')  # plus petites en premier
    for j in range(0,K):
        if distances[j][1]==1.:
            counter1+=1
        if distances[j][1]==2.:
            counter2+=1
        if counter1+counter2==K:
            break
    if counter1>counter2:
        return 1.
    else:
        return 2.

def predict(train, test, K):
    cdef x_test = test[:,1:]
    cdef N_test = test.shape[0]
    predictions = np.zeros(N_test, dtype=float)
    for i in range(N_test):
        predictions[i] = classifier(train, test, K, x_test)
    return predictions
