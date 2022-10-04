import numpy as np

def gradient2D(X:np.array)->tuple:
    """This function computes the 2D discrete gradient operator D applied to a matrix X of dimensions 2

    Args:
        X (array): A matrix in C^(M,N)

    Returns:
        (array,array) : A tuple in C^(M,N) x C^(M,N)
    """
    assert X.ndim <=2, "The input array has more than 2 dimensions"

    XDh = np.zeros(X.shape)
    for n in range(1,XDh.shape[1]):
        XDh[:,n-1] = X[:,n]-X[:,n-1]

    DvX = np.zeros(X.shape)
    for m in range(1,XDh.shape[0]):
        DvX[m-1,:] = X[m,:]-X[m-1,:]


    return XDh, DvX


def tv(X:np.array)->float:
    """This function compute the discrete isotropic total variation of an input matrix in C^(M,N)

    Args:
        X (np.array): A matrix in C^(MxN)

    Returns:
        float: returns the value of the TV for the input matrix X
    """
    XDh, DvX = gradient2D(X)
    sum = 0
    for m in range(XDh.shape[0]):
        for n in range(XDh.shape[1]):
            sum+= np.sqrt(XDh[m,n]**2 + DvX[m,n]**2)
    return sum


def gradient2D_adjoint(Y:tuple)->np.array:
    """This function computes the adjoint of the 2D discrete gradient operator D applied to a couple of matrices of dimension 2

    Args:
        Y (array,array): A tuple in C^(M,N) x C^(M,N)

    Returns:
        array: A matrix in C^(M,N)
    """
    (Yh, Yv) = Y
    YhDh = np.zeros(Yh.shape)
    DvYv = np.zeros(Yv.shape)

    YhDh[:,0] = - Yh[:,0]
    YhDh[:,-1] = Yh[:,-2]
    for n in range(1,Yh.shape[1]-1):
        YhDh[:, n] = -(Yh[:,n]- Yh[:,n-1])

    DvYv[0,:] = - Yv[0, :]
    DvYv[-1,:] =  Yv[-2, :]
    for m in range(1, Yv.shape[0]-1):
        DvYv[m,:] = - (Yv[m, :]  - Yv[m-1, :])

    D_adjoint = YhDh + DvYv
    return D_adjoint
