import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import sys
import os
import torch.sparse as sparse
from scipy.sparse import csr_matrix
import scipy.sparse
import scipy.sparse.linalg
import math
import numpy as np
import scipy 
import scipy.io

def GTS_RK2_onestep(U, dt, Nx, Ny, Nt, hx, hy, LF, limiter):
    U1 = torch.zeros_like(U)
    U2 = torch.zeros_like(U)
    
    col_size = U.shape[1]
#     x_mid=(x_grid[0:-1]+x_grid[1:])/2
    
    rhs0 = torch.zeros_like(U)
    rhs1 = torch.zeros_like(U)
    ## stage 1    
    for ll in range(col_size):
        rhs0[:, [ll]] += F_RHS(U[:, [ll]],Nx,Ny,hx,hy,LF)
        a = U[:, ll]
        b = rhs0[:, [ll]]

        U1[:, [ll]] += U[:, [ll]]+dt*rhs0[:, [ll]].clone()
        U1[:, [ll]] = slope_limiter(U1[:, [ll]], Nx, Ny, hx, hy, limiter)

        rhs1[:, [ll]] = F_RHS(U1[:, [ll]],Nx,Ny,hx,hy,LF)

        U2[:, [ll]] += (U[:, [ll]].clone())/2+(U1[:, [ll]].clone()+dt*rhs1[:, [ll]])/2

        U2[:, [ll]] = slope_limiter(U2[:, [ll]], Nx, Ny, hx, hy, limiter)
    return U2

def F_RHS(U,Nx,Ny,hx,hy,LF):
    U_s1,U_s2,U_n1,U_n2,U_w1,U_w2,U_e1,U_e2, V_s1,V_s2,V_n1,V_n2,V_w1,V_w2,V_e1,V_e2 = gauss_point(U,Nx,Ny)
    
    h_s1 = h2(U_s1,V_s1,LF)
    h_s2 = h2(U_s2,V_s2,LF)
    h_n1 = h1(U_n1,V_n1,LF)
    h_n2 = h1(U_n2,V_n2,LF)
    h_w1 = h2(U_w1,V_w1,LF)
    h_w2 = h2(U_w2,V_w2,LF)
    h_e1 = h1(U_e1,V_e1,LF)
    h_e2 = h1(U_e2,V_e2,LF)

    ff = torch.zeros_like(U)
    
    ff[0::3, :] = -0.5/hy*(h_s1+h_s2+h_n1+h_n2)-0.5/hx*(h_w1+h_w2+h_e1+h_e2)            # west, east
    
    U00 = U[0::3, :]
    U10 = U[1::3, :]
    U01 = U[2::3, :]
    
    FU = 3*U00**2+U10**2+U01**2;
    ff[1::3, :] = FU/hx -1.5 / hy*(-h_s1+h_s2-h_n1+h_n2) / torch.sqrt(torch.tensor(3.0))-1.5 / hx*(-h_w1-h_w2+h_e1+h_e2)            # phi^{10}
    ff[2::3, :] = FU/hy-1.5/hy*(-h_s1-h_s2+h_n1+h_n2)-1.5/hx*(-h_w1+h_w2-h_e1+h_e2) / torch.sqrt(torch.tensor(3.0))
    return ff

def gauss_point(U,Nx,Ny):
    #evaluate U at 8 Gaussian points each rectangle
    #s,n,w,e: south, north, west, east
    U00 = U[0::3, :]
    U10 = U[1::3, :]
    U01 = U[2::3, :]
    U_s1 = U00 - U10/torch.sqrt(torch.tensor(3.0)) - U01
    U_s2 = U00 + U10/torch.sqrt(torch.tensor(3.0)) - U01
    U_n1 = U00 - U10/torch.sqrt(torch.tensor(3.0)) + U01
    U_n2 = U00 + U10/torch.sqrt(torch.tensor(3.0)) + U01
    U_w1 = U00 - U10 - U01/torch.sqrt(torch.tensor(3.0))
    U_w2 = U00 - U10 + U01/torch.sqrt(torch.tensor(3.0))
    U_e1 = U00 + U10 - U01/torch.sqrt(torch.tensor(3.0))
    U_e2 = U00 + U10 + U01/torch.sqrt(torch.tensor(3.0))
   
    V_s1 = shift_below(U_n1,Nx,Ny)             # from U_n1
    V_s2 = shift_below(U_n2,Nx,Ny)             # from U_n2
    V_n1 = shift_above(U_s1,Nx,Ny)             # from U_s1 
    V_n2 = shift_above(U_s2,Nx,Ny)             # from U_s2
    V_w1 = shift_left(U_e1,Ny)                 # from U_e1
    V_w2 = shift_left(U_e2,Ny)                 # from U_e2
    V_e1 = shift_right(U_w1,Ny)                # from U_w1
    V_e2 = shift_right(U_w2,Ny)                # from U_w2
    
    V_s1 = V_s1[:, None].clone()
    V_s2 = V_s2[:, None].clone()
    V_n1 = V_n1[:, None].clone()
    V_n2 = V_n2[:, None].clone()
    return U_s1,U_s2,U_n1,U_n2,U_w1,U_w2,U_e1,U_e2, V_s1,V_s2,V_n1,V_n2,V_w1,V_w2,V_e1,V_e2

def f(x):
    ff = x**2/2

    return ff

def Df(x):
    ff = x

    return ff

def h1(u, v, LFG):
    if LFG == 1:
        alpha = 3 / 4
        hh = 0.5 * (f(u) + f(v) + alpha * (u - v))
    elif LFG == 2:
        zero_tensor = torch.zeros_like(u)
        hh = f(torch.max(u, zero_tensor))+f(torch.min(v, zero_tensor))
    return hh

def h2(u, v, LFG):
    if LFG == 1:
        alpha = 3/4
        hh = 0.5*(-f(u)-f(v) +alpha*(u-v))
    elif LFG  == 2:
        zero_tensor = torch.zeros_like(u)  # Ensure correct dtype and device
        hh = -f(torch.min(u, zero_tensor)) - f(torch.max(v, zero_tensor))
    return hh

                            
def shift_below(U, Nx, Ny):
    """Shift U down (periodic) in a (Ny, Nx) reshaped grid."""
    V = U.view(Ny, Nx)  # Reshape to (Ny, Nx)
    V = torch.cat((V[-1:, :], V[:-1, :]), dim=0)  # Shift rows down
    return V.reshape(-1)  # Flatten back to a vector

def shift_above(U, Nx, Ny):
    """Shift U up (periodic) in a (Ny, Nx) reshaped grid."""
    V = U.view(Ny, Nx)  # Reshape to (Ny, Nx)
    V = torch.cat((V[1:, :], V[:1, :]), dim=0)  # Shift rows up
    return V.reshape(-1)  # Flatten back to a vector

def shift_left(U, Ny):
    """Shift U left (periodic) for a vectorized column-major storage."""
    return torch.cat((U[-Ny:], U[:-Ny]))  # Last Ny elements wrap around

def shift_right(U, Ny):
    """Shift U right (periodic) for a vectorized column-major storage."""
    return torch.cat((U[Ny:], U[:Ny]))

def slope_limiter(U,Nx,Ny,hx,hy, limiter):
    Umod = U.clone()
    M = 50
    i = U.shape[1]
    if limiter == 1:
        for l in range(i):
            U00 = U[0::3, l]
            U10 = U[1::3, l]
            U01 = U[2::3, l]

            Ur = shift_right(U00,Ny)
            Ul = shift_left(U00,Ny)
            Ua = shift_above(U00,Nx,Ny)
            Ub = shift_below(U00,Nx,Ny) 
            
            W10 = torch.cat((U10[:, None], (Ur-U00)[:, None], (U00-Ul)[:, None]), dim=1)
#             print(W10.shape)
            W10 = torch.transpose(W10, 0, 1)

            Umod[1::3, l] = minmodB(W10,M,hx)
            W01 = torch.cat((U01[:, None], (Ua-U00)[:, None], (U00-Ub)[:, None]), dim = 1)
            W01 = torch.transpose(W01, 0, 1)

            Umod[2::3, l] = minmodB(W01,M,hy)
        
    return Umod
    
def minmodB(W, M, h):
    Ulim = W[0, :].clone()  # First row of W
    ids = torch.nonzero(torch.abs(Ulim) > M * h**2, as_tuple=True)[0]

    if ids.numel() > 0:
        Ulim[ids] = 0
        nrow = W.shape[0]
        s = torch.sum(torch.sign(W), dim=0) / nrow  # Compute s as in MATLAB
        ids2 = torch.nonzero(torch.abs(s) == 1, as_tuple=True)[0]
        ids_cap = torch.tensor(list(set(ids.tolist()) & set(ids2.tolist())), device=W.device)

        if ids_cap.numel() > 0:
            Ulim[ids_cap] = s[ids_cap] * torch.min(torch.abs(W[:, ids_cap]), dim=0)[0]

    return Ulim


def ReOrderSol(u, Nx, Ny):
    u_reorder = torch.zeros_like(u)
    u_reorder[:, :Nx*Ny] = u[:, 0::3].clone()
    u_reorder[:, Nx*Ny+torch.arange(0, Nx*Ny)] = u[:, 1::3].clone()
    u_reorder[:, 2*Nx*Ny+torch.arange(0, Nx*Ny)] = u[:, 2::3].clone()
    return u_reorder

def ReverseOrderSol(u, Nx, Ny):
    u_reverse = torch.zeros_like(u)
    
    u_reverse[:, 0::3] = u[:, :Nx*Ny]
    u_reverse[:, 1::3] = u[:, Nx*Ny+torch.arange(0, Nx*Ny)]
    u_reverse[:, 2::3] = u[:, 2*Nx*Ny+torch.arange(0, Nx*Ny)]
    return u_reverse