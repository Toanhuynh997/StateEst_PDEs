import numpy as np
import scipy
from scipy import sparse
import scipy.sparse as sp
from scipy.sparse import eye, diags, block_diag, hstack
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import gmres, spsolve
from scipy.sparse import eye, diags, kron, csr_matrix


def Mobility_NonCons(x, gamma, cons, noise):
    if cons == 1:
        return np.ones((x.shape[0], ))
    elif cons == 2:
        return np.ones((x.shape[0], ))+noise
    elif cons == 0:
        return 1-(gamma**2)*(x**2)
    elif cons == 3:
        return np.maximum(np.ones((x.shape[0], ))+noise, 1)
    elif cons == 4:
        return np.maximum(1-x**2+noise, 0)
    elif cons == 5:
        return (1-(x**2))*np.exp(noise)
    elif cons == 6:
        return np.maximum(np.ones((x.shape[0], ))+noise, 0)
    
def f_doublewell(x):
    return x**3-x

def AllenCahn_Solver_1step_BDF1(U0, N, tau, eps, gamma, S, Dh, cons, noise):
    MU = Mobility_NonCons(U0, gamma, cons, noise)
#     print(MU)
#     Lamb = np.diag(MU)
#     Mat = np.eye(N**2)-(tau*(eps**2)*Lamb) @ Dh + tau*S*np.eye(N**2)
    Lamb = diags(MU, 0, format='csr')
    I2 = eye(N**2, format='csr')
    Mat_sparse = (I2- tau*(eps**2)*(Lamb@ Dh)+ tau*S*I2).tocsr()
    
    RHS = U0+tau*S*U0-tau*(Lamb @ f_doublewell(U0))
#     Mat_sparse = sparse.csr_matrix(Mat)
    x = spsolve(Mat_sparse, RHS)
    
    return x


def AllenCahn_Solver_1step_BDF2(U1, U0, N, tau, eps, gamma, S, Dh, b0, b1, cons, noise):
#     Lamb = np.diag(MUbar)
    
#     Mat = b0*np.eye(N**2)-((eps**2)*Lamb) @ Dh+S*np.eye(N**2)
    xbar = AllenCahn_Solver_1step_BDF1(U1, N, tau, eps, gamma, S, Dh, cons, noise)
    MUbar = Mobility_NonCons(xbar, gamma, cons, noise)
    Lamb = diags(MUbar, 0, format='csr')
    
    I2 = eye(N**2, format='csr')
    Mat_sparse = (b0*I2-((eps**2)*Lamb) @ Dh+S*I2).tocsr()
    
    RHS = (b0-b1)*U1+b1*U0-(Lamb @ f_doublewell(xbar))+S*xbar
    
    x = spsolve(Mat_sparse, RHS)
    
    return x
    
    