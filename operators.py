from numpy import arange
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

def blur_matrix(n, m):
    """
    Construct an elementary sparse blur matrix via nearest-neighbor
    averaging.
    """

    ### Your code here ###

    N = n * m

    rows = []
    cols = []
    data = []

    for i in range(n):
        for j in range(m):
            # Index of the current pixel in the flattened array
            index = i * m + j

            # Central pixel
            rows.append(index)
            cols.append(index)
            data.append(1/2)

            # Right neighbor
            if j + 1 < m:
                rows.append(index)
                cols.append(index + 1)
                data.append(1/8)

            # Left neighbor
            if j - 1 >= 0:
                rows.append(index)
                cols.append(index - 1)
                data.append(1/8)

            # Lower neighbor
            if i + 1 < n:
                rows.append(index)
                cols.append(index + m)
                data.append(1/8)

            # Upper neighbor
            if i - 1 >= 0:
                rows.append(index)
                cols.append(index - m)
                data.append(1/8)

    # Create the sparse matrix in CSR format
    blur_matrix = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return blur_matrix

def tychonov_matrices(n, m, radius=1, alpha=0.1):
    ### Your code here ###
    B = (blur_matrix(n, m))**radius
    # Calculate BT
    BT = B.transpose()
    # Calculate M
    # M = (B^T B + alpha*I) x_alpha
    M = BT.dot(B) + alpha * sp.eye(B.shape[0])
    return M, BT
    
def tychonov_operators(n, m, radius=1, alpha=0.1):
    ### Your code here ###
    N = n * m
    B = blur_matrix(n, m)**radius

    # Define matvec for M operator
    def Mvec(x):
        return (B.transpose()).dot(B.dot(x)) + alpha * x

    # Define matvec for BT operator
    def BTvec(x):
        return (B.dot(x).transpose())
    
    return (LinearOperator((N,N), matvec=Mvec),
            LinearOperator((N,N), matvec=BTvec))


