import numpy as np

def gradient2D(X:np.array)->tuple:
    """This function computes the 2D discrete gradient operator D applied to a matrix X of dimensions 1 or 2

    Args:
        X (np.array):A matrix in C^(M,N)

    Returns:
        (np.array,np.array) : Two matrix in C^(M,N)
    """
    assert X.ndim <=2, "The input array has more than 2 dimensions"

    XDh = np.zeros(X.shape)
    for n in range(2,XDh.shape[1]):
        XDh[:,n-2] = X[:,n]-X[:,n-1]

    DvX = np.zeros(X.shape)
    for m in range(2,XDh.shape[0]):
        DvX[m-2,:] = X[m,:]-X[m-1,:]


    return XDh, DvX


def tv(X:np.array)->float:
    """This function compute the discrete isotropic total variation of an input matrix in C^(M,N)

    Args:
        X (np.array): A matrix in C^(MxN)

    Returns:
        float: returns the value of the TV for the input matrix X
    """
    DX = gradient2D(X)
    sum = 0
    for m in range(DX.shape[0]):
        for n in range(DX.shape[1]):
            sum+= np.sqrt(DX[m,n,0]**2 + DX[m,n,1]**2)
    return sum


def gradient2D_adjoint(Y:np.array)->np.array:
    """_summary_

    Args:
        Y (np.array): _description_

    Returns:
        np.array: _description_
    """
    (Yh, Yv) = Y
    YhDh = np.zeros(Yh.shape)
    DvYv = np.zeros(Yv.shape)

    YhDh[:,0] = - Yh[:,1]
    YhDh[:,-1] =  Yh[:,-1]
    for n in range(2,Yh.shape[1]):
        YhDh[:, n-1] = -(Yh[:,n]- Yh[:,n-1])

    DvYv[0,:] = - Yv[1, :]
    DvYv[-1,:] =  Yv[-1, :]
    for m in range(2, Yv.shape[0]):
        DvYv[m-1,:] = - (Yv[m, :]  - Yv[m-1, :])

    D_adjoint =YhDh + DvYv
    return D_adjoint
