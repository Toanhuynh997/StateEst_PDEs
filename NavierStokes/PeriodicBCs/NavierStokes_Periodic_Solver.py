import numpy as np
import scipy.sparse as sp
from scipy.sparse import eye, diags, block_diag, hstack
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla

def avg(A):
    """
    Computes the average of consecutive entries of A along the first dimension.
    If A is a row vector (or a 1D array), averages consecutive elements.
    Otherwise, for a 2D array with more than one row, averages consecutive rows.
    """
    A = np.asarray(A)
    # Check if A is 1D or a 2D row vector.
    if A.ndim == 1 or (A.ndim == 2 and A.shape[0] == 1):
        a = A.ravel()  # flatten to 1D if needed
        return (a[1:] + a[:-1]) / 2.0
    else:
        return (A[1:, :] + A[:-1, :]) / 2.0
    
    
def inner(U, V):
    """
    Computes the inner product: sum(sum(U.*V))
    """
    return np.sum(U * V)


def compute_UgradU(U, V, hx, hy):
    """
    U, V: periodic BCs
    """
    U = np.asarray(U)
    V = np.asarray(V)
    
    Udx, Udy = grad_velo(U,hx,hy)
    Vmod = np.vstack([V[-1, :], V])
    Vmod = np.hstack([Vmod, Vmod[:,0][:, None]])
    
    Vmod = avg(Vmod)
    Vmod = avg(Vmod.T).T;
    UgradU1 = U*Udx + Vmod*Udy;

    Vdx, Vdy = grad_velo(V,hx,hy)
    Umod = np.hstack([U[:,-1][:, None], U])
    Umod = np.vstack([Umod, Umod[0,:]])
    
    Umod = avg(Umod)
    Umod = avg(Umod.T).T
    UgradU2 = Umod*Vdx + V*Vdy;                            
    
    return UgradU1, UgradU2


def grad_velo(U,hx,hy):
    Udx = np.vstack([U[-1,:], U, U[0,:]])
    Udx = (Udx[2:,:]-Udx[:-2,:]) / (2*hx) # Udx~U
    
    Udy = np.hstack([U[:,-1][:, None], U, U[:,0][:, None]])   
    Udy = (Udy[:,2:]-Udy[:,:-2]) / (2*hy) # Udy~U
    return Udx, Udy

def DiscreteGrad(N,h):
    # size(A) = [N,N], periodic
    diagonals = [-np.ones(N), np.ones(N)]
    offsets = [-1, 0]
    A = sp.diags(diagonals, offsets, shape=(N, N)) / h
    A = A.tolil()  # Convert to LIL for easy modification
    A[0, -1] = -1 / h  # Set periodic boundary condition
    return A.tocsr()

def DiscreteLaplace(N,h):
    # periodic
    diagonals = [np.ones(N), -2 * np.ones(N), np.ones(N)]
    offsets = [-1, 0, 1]
    A = sp.diags(diagonals, offsets, shape=(N, N)) / h**2
    A = A.tolil()  # Convert to LIL for easy modification
    A[0, -1] = 1 / h**2  # Periodic boundary at (1, N)
    A[-1, 0] = 1 / h**2  # Periodic boundary at (N, 1)
    return A.tocsr()  # Convert back to CSR for efficiency


def stokes_solver(f, DiE, BDiE, DiBt, Di, B, Bt, perS, SS, SSt, Nx, Ny, sU):
    """
    periodic BCs
    """
    # Define the lhs operator using the helper function lhs_u.
   
    def lhs(u):
        return lhs_u(u, DiE, BDiE, DiBt, perS, SS, SSt)
    
    M_Lhs = spla.LinearOperator(DiE.shape, lhs)

    rhs_1 = Di @ f
    rhs_2 = B @ rhs_1
    
    
    # Solve SS \ (SSt \ rhs_2[perS])
    temp = np.linalg.solve(SSt, rhs_2[perS])
    rhs_2[perS] = np.linalg.solve(SS, temp)
    rhs = rhs_1 - Di @ Bt @ rhs_2
    
#     print(rhs)
    #     print(rhs)
    u, info = gmres(M_Lhs, rhs, tol=1e-8, maxiter=1000)
    if info != 0:
        print("GMRES did not converge, info =", info)
    
#     print(u.shape)
    p = BDiE @ u
    temp = np.linalg.solve(SSt, p[perS])
    p[perS] = np.linalg.solve(SS, temp)
    p = p + rhs_2
    
#     print(u[:sU])
#     print(u[sU:])
    U = np.reshape(u[:sU], (Nx, Ny), order='F')
    V = np.reshape(u[sU:], (Nx, Ny), order='F')
    
    # Prepend a zero to p and remove the mean before reshaping.
    p_ex = np.concatenate(([0], p))
    P = np.reshape(p_ex - np.mean(p_ex), (Nx, Ny), order='F')
    
    return U, V, P


def lhs_u(u, DiE, BDiE, DiBt, perS, SS, SSt):
    """
    periodic BCs
    """    
    lhs_val = BDiE @ u
 
    temp = np.linalg.solve(SSt, lhs_val[perS])
    lhs_val[perS] = np.linalg.solve(SS, temp)
    return u - (DiE @ u) + (DiBt @ lhs_val)


def u_init(x, y, opt):
    """
    Exact u-velocity.
    """
    if opt == 1:
        return np.sin(x)*np.cos(y)
    elif opt == 2:        
        return np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    
    
def v_init(x, y, opt):
    """
    Exact u-velocity.
    """
    if opt == 1:
        return -np.cos(x)*np.sin(y)
    elif opt == 2:        
        return -np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    
def u_exact(x, y, t, mu, opt):
    """
    Exact u-velocity.
    """
    if opt == 1:
        return np.sin(x)*np.cos(y)*np.exp(-2*mu*t)
    elif opt == 2:        
        return np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*np.exp(-8*np.pi^2*mu*t)

def v_exact(x, y, t, mu, opt):
    """
    Exact v-velocity.   
    """
    if opt == 1:
        return -np.cos(x)*np.sin(y)*np.exp(-2*mu*t)
    elif opt == 2:
        return -np.cos(2*np.pi*x)*np.sin(2*np.pi*y)*np.exp(-8*np.pi^2*mu*t)


def p_exact(x, y, t, mu, opt):
    """
    Exact pressure.
    """
    if opt == 1:
        return 0.25*(np.cos(2*x)+np.cos(2*y))*np.exp(-4*mu*t)
    elif opt == 2:
        return 0.25*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*np.exp(-16*np.pi^2*mu*t)

def f1(x, y, t, mu, opt):
    """
    Right-hand side forcing term f1.    
    """
    if opt in (1, 2):        
        return 0*x+0*y+0*t+0*mu

def f2(x, y, t, mu, opt):
    """
    Right-hand side forcing term f2.
    """
    if opt in (1, 2): 
        return 0*x+0*y+0*t+0*mu
    
    
#### First-order 
def NS_BE_1step_Periodic(hx, hy, dt, TT, U, V, q, Xu, Yu, Xv, Yv, mu, theta, opt, opt_UgradU, DiE, BDiE, DiBt, Di,\
                         B, Bt,  perS, SS, SSt, Nx, Ny, sU, alpha, A1, B1):                   
    # Step 1: find U1, V1, P1
    ff1 = f1(Xu, Yu, TT, mu, opt)          # Xu, Yu defined via meshgrid (matching MATLAB dimensions)    
    ff2 = f2(Xv, Yv, TT, mu, opt)

    rhs1 = alpha * U + ff1
    rhs2 = alpha * V + ff2  
    
    # Form the right-hand side as a concatenated 1D vector:
    rhs1_flat = np.reshape(rhs1, rhs1.shape[0]*rhs1.shape[1], order='F')
    rhs2_flat = np.reshape(rhs2, rhs2.shape[0]*rhs2.shape[1], order='F') 
    rhs = np.concatenate([rhs1_flat, rhs2_flat])
    
   
    U1, V1, P1 = stokes_solver(rhs, DiE, BDiE, DiBt, Di, B, Bt,
                               perS, SS, SSt, Nx, Ny, sU)

    # Step 2: find U2, V2, P2 using the computed nonlinear term
 
    UgradU1, UgradU2 = compute_UgradU(U, V, hx, hy)
  
    UgradU1_flat = np.reshape(UgradU1, UgradU1.shape[0]*UgradU1.shape[1], order='F')
    UgradU2_flat = np.reshape(UgradU2, UgradU2.shape[0]*UgradU2.shape[1], order='F')    
    rhs_grad = -np.concatenate([UgradU1_flat, UgradU2_flat])

    U2, V2, P2 = stokes_solver(rhs_grad , DiE, BDiE, DiBt, Di, B, Bt,
                               perS, SS, SSt, Nx, Ny, sU)   
  
    # Step 3: find q by computing several spatial derivatives
    U2dx = (-A1.T)*U2
    U2dy = U2*(-B1)
    V2dx = (-A1.T)*V2
    V2dy = V2*(-B1);
    
    # Compute coefficients for the quadratic equation in q.
    a = 0.5 * (inner(U2, U2) + inner(V2, V2)) + \
         dt * mu * (inner(U2dx, U2dx) + inner(U2dy, U2dy) + \
                    inner(V2dx, V2dx) + inner(V2dy, V2dy)) 
    b = (inner(U - U1, U2) + inner(V - V1, V2) - \
         dt * (inner(UgradU1, U1) + inner(UgradU2, V1)))
    c = -0.5 * (inner(U1 - U, U1 - U) + inner(V1 - V, V1 - V))

    # Scale by grid cell area and add the theta-term (theta assumed defined)
    a_new = a * hx * hy + theta
    b_new = b * hx * hy
    c_new = c * hx * hy - q**2 * theta
    
    # Solve the quadratic for q (taking the positive square root branch)
    q_new = (-b_new + np.sqrt(b_new**2 - 4 * a_new * c_new)) / (2 * a_new)

    # Step 4: update U and V with the new correction.
    U_new = U1 + q_new * U2
    V_new = V1 + q_new * V2
    egy = 0.5 * hx * hy * (inner(U_new, U_new) + inner(V_new, V_new))
    qq = q_new
    egy_theta = egy+theta*(q_new**2-1)
           
    # After the loop, update pressure:
    P_new = P1 + q_new * P2
    return U_new, V_new, P_new, q_new, egy, egy_theta, qq


## BDF2 Periodic
def NS_BDF2_1step_periodic(hx, hy, dt, TT, U_old_old, V_old_old, U_old, V_old, q_old, q_old_old, Xu, Yu, Xv, Yv, mu,\
                           theta, opt, opt_UgradU, DiE, BDiE, DiBt, Di, B, Bt,  perS, SS, SSt, Nx, Ny, sU, alpha, A1, B1):
                    
#     for k in range(Nt):
    # Step 1: find U1, V1, P1
    ff1 = f1(Xu, Yu, TT, mu, opt)          # Xu, Yu defined via meshgrid (matching MATLAB dimensions)
    rhs1 = (4*U_old-U_old_old)/(2*dt) + ff1
    ff2 = f2(Xv, Yv, TT, mu, opt)
    rhs2 = (4*V_old-V_old_old)/(2*dt) + ff2
  
    # Form the right-hand side as a concatenated 1D vector:
    rhs1_flat = np.reshape(rhs1, rhs1.shape[0]*rhs1.shape[1], order='F')
    rhs2_flat = np.reshape(rhs2, rhs2.shape[0]*rhs2.shape[1], order='F') 
    rhs = np.concatenate([rhs1_flat, rhs2_flat])   

  
    U1, V1, P1 = stokes_solver(rhs, DiE, BDiE, DiBt, Di, B, Bt,
                               perS, SS, SSt, Nx, Ny, sU)
  
    

    # Step 2: find U2, V2, P2 using the computed nonlinear term
   
    UgradU1, UgradU2 = compute_UgradU(2*U_old-U_old_old, 2*V_old-V_old_old, hx, hy)
   
    # Build the right-hand side for U2/V2 (note the minus sign)
    
#     rhs = -np.concatenate([UgradU1.flatten(), UgradU2.flatten()])
    
    UgradU1_flat = np.reshape(UgradU1, UgradU1.shape[0]*UgradU1.shape[1], order='F')
    UgradU2_flat = np.reshape(UgradU2, UgradU2.shape[0]*UgradU2.shape[1], order='F') 
    
    rhs_grad = -np.concatenate([UgradU1_flat, UgradU2_flat])
    
    
    U2, V2, P2 = stokes_solver(rhs_grad , DiE, BDiE, DiBt, Di, B, Bt,
                               perS, SS, SSt, Nx, Ny, sU)

    # Step 3: find q by computing several spatial derivatives
    U2dx = (-A1.T)*U2
    U2dy = U2*(-B1)
    V2dx = (-A1.T)*V2
    V2dy = V2*(-B1)
    
    # Compute coefficients for the quadratic equation in q.
    a = 1.5*(inner(U2,U2)+inner(V2,V2))+2*dt*mu*(inner(U2dx,U2dx)+inner(U2dy,U2dy)+\
                                                 inner(V2dx,V2dx)+inner(V2dy,V2dy))
    
    b = -inner(3*U1-4*U_old+U_old_old,U2)-inner(3*V1-4*V_old+V_old_old,V2)-2*dt*(inner(UgradU1,U1)+inner(UgradU2,V1))
    c = -2*(inner(U1-U_old,U1-U_old)+inner(V1-V_old,V1-V_old))\
            +0.5*(inner(U1-U_old_old,U1-U_old_old)+inner(V1-V_old_old,V1-V_old_old))       
    
    # Scale by grid cell area and add the theta-term (theta assumed defined)
   
    a_new = a * hx * hy + 3*theta
    b_new = b * hx * hy
    c_new = c*hx*hy-theta*(4*q_old**2-q_old_old**2)
      
    # Solve the quadratic for q (taking the positive square root branch)
    delta = b_new**2-4*a_new*c_new
    q1_new = (-b_new+np.sqrt(delta))/(2*a_new)
    q2_new = (-b_new-np.sqrt(delta))/(2*a_new)
    
    if abs(q1_new-1)<abs(q2_new-1):
        q_new = q1_new 
    else:
        q_new = q2_new
    
#     print(q_new)
    # Step 4: update U and V with the new correction.
    U_new = U1 + q_new * U2
    V_new = V1 + q_new * V2
    egy = 0.5 * hx * hy * (inner(U_new, U_new) + inner(V_new, V_new))
#     qq = q_new

    # Check the energy equation:
    err_egy = (hx * hy * (inner(U_new, U_new) + inner(V_new, V_new) - inner(U_old, U_old) - inner(V_old, V_old)) / (2 * dt) +
               theta / dt * (q_new**2 - q_old**2) -
               hx * hy * (inner((U_new - U_old) / dt + q_new * UgradU1, U_new) +
                           inner((V_new - V_old) / dt + q_new * UgradU2, V_new)))
#     print("Error in the energy equation: {:.2e}".format(err_egy))

    # After the loop, update pressure:
    P_new = P1 + q_new * P2
    return U_new, V_new, P_new, q_new, egy


##### Vectorization part after this notice

def avg_batch(A):
    """
    A version of the 'avg' function that handles up to 3D arrays without using ellipsis.
    
    Behavior:
      1D (shape=(n,)):
        - Average consecutive elements -> returns shape (n-1,)
        
      2D:
        - If shape=(1, n): Flatten to 1D, then average consecutive elements.
        - Else shape=(m, n) with m>1: Average consecutive rows along axis=0 -> returns shape (m-1, n)
        
      3D:
        - If shape=(1, n, l): Flatten everything to 1D of length n*l, then average consecutive elements.
        - Else shape=(m, n, l) with m>1: Average consecutive “rows” along axis=0 -> returns shape (m-1, n, l)
    """
    A = np.asarray(A)
    
    # ---------------------
    # 1D case: shape (n,)
    # ---------------------
    if A.ndim == 1:
        # Average consecutive 1D elements
        return 0.5 * (A[1:] + A[:-1])
    
    # ---------------------
    # 2D case: shape (m, n)
    # ---------------------
    elif A.ndim == 2:
        m, n = A.shape
        if m == 1:
            # Flatten row vector to 1D and average
            flat = A.ravel()  # shape (n,)
            return 0.5 * (flat[1:] + flat[:-1])
        else:
            # Average consecutive rows -> shape (m-1, n)
            return 0.5 * (A[1:, :] + A[:-1, :])
    
    # ---------------------
    # 3D case: shape (m, n, l)
    # ---------------------
    elif A.ndim == 3:
        m, n, l = A.shape
        if m == 1:
            # Flatten everything to 1D (length n*l) and average
            flat = A.ravel()
            return 0.5 * (flat[1:] + flat[:-1])
        else:
            # Average along axis=0 -> shape (m-1, n, l)
            return 0.5 * (A[1:, :, :] + A[:-1, :, :])
    
    # ---------------------
    # Otherwise, not implemented
    # ---------------------
    else:
        raise ValueError("avg() is only implemented for up to 3D arrays.")
            
def inner_batch(U, V):
    """
    Performs a batch inner product over all dimensions except the last one.
   
    """
    U = np.asarray(U)
    V = np.asarray(V)
    
    # We assume U and V have the same shape (..., L).
    if U.shape != V.shape:
        raise ValueError("U and V must have the same shape")
   
    return np.sum(U*V, axis=(0, 1))


def compute_UgradU_batch(U, V, hx, hy):
    """
    U, V: periodic BCs
    """
    U = np.asarray(U)
    V = np.asarray(V)
    
    Udx, Udy = grad_velo_batch(U,hx,hy)
    
    Vmod = np.concatenate([V[-1:, :, :], V], axis=0)
    Vmod = np.concatenate([Vmod, Vmod[:, :1, :]], axis=1)
    
#     print(Vmod[:, :, 0])
    Vmod = avg_batch(Vmod)

    Vmod = np.transpose(avg_batch(np.transpose(Vmod, (1, 0, 2))), (1, 0, 2))
    
#     print(Vmod[:, :, 0])
    
    UgradU1 = U*Udx + Vmod*Udy

    Vdx, Vdy = grad_velo_batch(V,hx,hy)
    Umod = np.concatenate([U[:, -1:, :], U], axis=1)
    Umod = np.concatenate([Umod, Umod[:1, :, :]], axis =0)
    
    Umod = avg_batch(Umod)
    Umod = np.transpose(avg_batch(np.transpose(Umod, (1, 0, 2))), (1, 0, 2))
    UgradU2 = Umod*Vdx + V*Vdy                          
    
    return UgradU1, UgradU2


def grad_velo_batch(U,hx,hy):
    Udx = np.concatenate([U[-1:, :, :], U, U[:1, :, :]], axis=0)
    Udx = (Udx[2:,:, :]-Udx[:-2,:, :]) / (2*hx) # Udx~U
    
    Udy = np.concatenate([U[:, -1:, :], U, U[:, :1, :]], axis=1)
    Udy = (Udy[:,2:, :]-Udy[:,:-2, :]) / (2*hy) # Udy~U
    return Udx, Udy

def stokes_solver_vectorized(f, DiE, BDiE, DiBt, Di, B, Bt, perS, SS, SSt, Nx, Ny, sU):
    """
    Vectorized version of stokes_solver to handle multiple RHS in 'f'.
    
    If f.ndim == 1, we do a single solve (same as original).
    If f.ndim == 2, we assume shape (n, L), i.e. L distinct RHS vectors.
    We'll compute the partial results in batch, then loop over columns for GMRES.
    
    Returns:
      U : shape (Nx-1, Ny, L) if L>1, or (Nx-1, Ny) if L=1
      V : shape (Nx, Ny-1, L) or (Nx, Ny-1)
      P : shape (Nx, Ny, L)    or (Nx, Ny)
    """

    f = np.asarray(f)
    # -- 1) Handle single-RHS vs multi-RHS --
    if f.ndim == 1:
        # Just do the classic single solve
        return stokes_solver(f, DiE, BDiE, DiBt, Di, B, Bt, perS, SS, SSt, Nx, Ny, sU)

    elif f.ndim == 2:
        # f.shape = (n, L)
        n, L = f.shape
        
        # Define the lhs operator only once
        def lhs(u):
            return lhs_u(u, DiE, BDiE, DiBt, perS, SS, SSt)
        
        # We'll convert 'lhs' into a LinearOperator for GMRES
        M_Lhs = spla.LinearOperator(DiE.shape, lhs)
        
        # -- 2) Vectorized pre-processing steps --
        #    Di, B, Bt, etc. can multiply all columns of f simultaneously
        #    shape logic:
        #      Di:    (n, n)
        #      B:     (n, n)
        #      Bt:    (n, n)
        #      f:     (n, L) => (n, L) after multiplication
        rhs_1 = Di @ f                     # shape (n, L)
        rhs_2 = B @ rhs_1                  # shape (n, L)
        
        # Now we solve for each column in rhs_2[perS, :] using SSt and SS
        # shape of rhs_2[perS, :] => (len(perS), L)
        temp = np.linalg.solve(SSt, rhs_2[perS, :])        # shape (len(perS), L)
        rhs_2[perS, :] = np.linalg.solve(SS, temp)         # shape (len(perS), L)
        
        # Then rhs => shape (n, L)
        rhs = rhs_1 - Di @ (Bt @ rhs_2)    # shape (n, L)
        
        # -- 3) Loop over each column, solve GMRES individually --
        U_list = []
        V_list = []
        P_list = []
        
        for i in range(L):
            # gmres wants a 1D vector, so we pass rhs[:, i]
            # The linear operator M_Lhs is the same for all
            sol, info = gmres(M_Lhs, rhs[:, i], tol=1e-8, maxiter=1000)
            if info != 0:
                print(f"GMRES did not converge for batch={i}, info={info}")
            
            # -- 4) Post-processing for p, U, V, P (per column) --
            p_i = BDiE @ sol                # shape (n,)
            temp = np.linalg.solve(SSt, p_i[perS])
            p_i[perS] = np.linalg.solve(SS, temp)
            p_i = p_i + rhs_2[:, i]
            
            # Extract velocity components
            U_i = sol[:sU]
            V_i = sol[sU:]
            
            # Reshape
            U_i = np.reshape(U_i, (Nx, Ny), order='F')
            V_i = np.reshape(V_i, (Nx, Ny), order='F')
            
            # Prepend zero to p_i
            p_ex = np.concatenate(([0], p_i))
            # Reshape => (Nx, Ny)
            P_i = np.reshape(p_ex - np.mean(p_ex), (Nx, Ny), order='F')
            
            U_list.append(U_i)
            V_list.append(V_i)
            P_list.append(P_i)
        
        # -- 5) Stack results along a new axis (the last dimension) --
        # shape of U => (Nx-1, Ny, L)
        U_out = np.stack(U_list, axis=-1)
        # shape of V => (Nx, Ny-1, L)
        V_out = np.stack(V_list, axis=-1)
        # shape of P => (Nx, Ny, L)
        P_out = np.stack(P_list, axis=-1)
        
        return U_out, V_out, P_out
    
    else:
        raise ValueError("f must be 1D or 2D (batch), but got f.ndim={}".format(f.ndim))
        
#### First-order 
def NS_BE_1step_Periodic_Vectorized(hx, hy, dt, TT, U, V, q, Xu, Yu, Xv, Yv, mu, theta, opt, opt_UgradU, DiE, BDiE, DiBt, Di,\
                                    B, Bt,  perS, SS, SSt, Nx, Ny, sU, alpha, A1, B1):                   
    # Step 1: find U1, V1, P1
    ff1 = f1(Xu, Yu, TT, mu, opt)          # Xu, Yu defined via meshgrid (matching MATLAB dimensions)    
    ff2 = f2(Xv, Yv, TT, mu, opt)

    rhs1 = alpha * U + ff1[:, :, np.newaxis]    
    rhs2 = alpha * V + ff2[:, :, np.newaxis]      
    
    # Form the right-hand side as a concatenated 1D vector:
    rhs1_flat = np.reshape(rhs1, (rhs1.shape[0]*rhs1.shape[1],  -1), order='F')
    rhs2_flat = np.reshape(rhs2, (rhs2.shape[0]*rhs2.shape[1],  -1), order='F')
    
    rhs = np.concatenate([rhs1_flat, rhs2_flat], axis=0)
    
   
    U1, V1, P1 = stokes_solver_vectorized(rhs, DiE, BDiE, DiBt, Di, B, Bt, perS, SS, SSt, Nx, Ny, sU)
   
    # Step 2: find U2, V2, P2 using the computed nonlinear term
   
    UgradU1, UgradU2 = compute_UgradU_batch(U, V, hx, hy)
#     print(UgradU1[:, :, 0])
    
    UgradU1_flat = np.reshape(UgradU1, (UgradU1.shape[0]*UgradU1.shape[1], -1), order='F')
    UgradU2_flat = np.reshape(UgradU2, (UgradU2.shape[0]*UgradU2.shape[1], -1), order='F')
    rhs_grad = -np.concatenate([UgradU1_flat, UgradU2_flat], axis=0)

    U2, V2, P2 = stokes_solver_vectorized(rhs_grad , DiE, BDiE, DiBt, Di, B, Bt, perS, SS, SSt, Nx, Ny, sU)   
  
    # Step 3: find q by computing several spatial derivatives
    U2dx =  (-A1.T).toarray()[:, :, np.newaxis]*U2
  
    U2dy = U2*(-B1.toarray()[:, :, np.newaxis])
    V2dx =  (-A1.T).toarray()[:, :, np.newaxis]*V2
    V2dy = V2*(-B1.toarray()[:, :, np.newaxis])
    
    # Compute coefficients for the quadratic equation in q.
    a = 0.5 * (inner_batch(U2, U2) + inner_batch(V2, V2)) + \
         dt * mu * (inner_batch(U2dx, U2dx) + inner_batch(U2dy, U2dy) + \
                    inner_batch(V2dx, V2dx) + inner_batch(V2dy, V2dy)) 
    b = (inner_batch(U - U1, U2) + inner_batch(V - V1, V2) - \
         dt * (inner_batch(UgradU1, U1) + inner_batch(UgradU2, V1)))
    c = -0.5 * (inner_batch(U1 - U, U1 - U) + inner_batch(V1 - V, V1 - V))

    # Scale by grid cell area and add the theta-term (theta assumed defined)
    a_new = a * hx * hy + theta
    b_new = b * hx * hy
    c_new = c * hx * hy - q**2 * theta
    
    # Solve the quadratic for q (taking the positive square root branch)
    q_new = (-b_new + np.sqrt(b_new**2 - 4 * a_new * c_new)) / (2 * a_new)

    # Step 4: update U and V with the new correction.
    q_new_resh = q_new.reshape(1, 1, -1)
    U_new = U1 + q_new_resh * U2
    V_new = V1 + q_new_resh * V2
    egy = 0.5 * hx * hy * (inner_batch(U_new, U_new) + inner_batch(V_new, V_new))
    qq = q_new
    egy_theta = egy+theta*(q_new_resh**2-1)
           
    # After the loop, update pressure:
    P_new = P1 + q_new * P2
    return U_new, V_new, P_new, q_new, egy, egy_theta, qq


## BDF2 Periodic
def NS_BDF2_1step_periodic_Vectorized(hx, hy, dt, TT, U_old_old, V_old_old, U_old, V_old, q_old, q_old_old, Xu, Yu, Xv, Yv, mu,\
                                      theta, opt, opt_UgradU, DiE, BDiE, DiBt, Di, B, Bt,  perS, SS, SSt, Nx, Ny, sU, alpha,\
                                      A1, B1):
                    
#     for k in range(Nt):
    # Step 1: find U1, V1, P1
    ff1 = f1(Xu, Yu, TT, mu, opt)          # Xu, Yu defined via meshgrid (matching MATLAB dimensions)
    rhs1 = (4*U_old-U_old_old)/(2*dt) + ff1[:, :, np.newaxis]   
    ff2 = f2(Xv, Yv, TT, mu, opt)
    rhs2 = (4*V_old-V_old_old)/(2*dt) + ff2[:, :, np.newaxis]   
  
    # Form the right-hand side as a concatenated 1D vector:
    rhs1_flat = np.reshape(rhs1, (rhs1.shape[0]*rhs1.shape[1],  -1), order='F')
    rhs2_flat = np.reshape(rhs2, (rhs2.shape[0]*rhs2.shape[1],  -1), order='F')
    
    rhs = np.concatenate([rhs1_flat, rhs2_flat], axis=0)

  
    U1, V1, P1 = stokes_solver_vectorized(rhs, DiE, BDiE, DiBt, Di, B, Bt, perS, SS, SSt, Nx, Ny, sU)
  
    # Step 2: find U2, V2, P2 using the computed nonlinear term
   
    UgradU1, UgradU2 = compute_UgradU_batch(2*U_old-U_old_old, 2*V_old-V_old_old, hx, hy)
   
    # Build the right-hand side for U2/V2 (note the minus sign)
    
    UgradU1_flat = np.reshape(UgradU1, (UgradU1.shape[0]*UgradU1.shape[1], -1), order='F')
    UgradU2_flat = np.reshape(UgradU2, (UgradU2.shape[0]*UgradU2.shape[1], -1), order='F')
    
    rhs_grad = -np.concatenate([UgradU1_flat, UgradU2_flat], axis=0)

    U2, V2, P2 = stokes_solver_vectorized(rhs_grad , DiE, BDiE, DiBt, Di, B, Bt, perS, SS, SSt, Nx, Ny, sU)

    # Step 3: find q by computing several spatial derivatives
    U2dx =  (-A1.T).toarray()[:, :, np.newaxis]*U2
  
    U2dy = U2*(-B1.toarray()[:, :, np.newaxis])
    V2dx =  (-A1.T).toarray()[:, :, np.newaxis]*V2
    V2dy = V2*(-B1.toarray()[:, :, np.newaxis])
    
    # Compute coefficients for the quadratic equation in q.
    a = 1.5*(inner_batch(U2,U2)+inner_batch(V2,V2))\
             +2*dt*mu*(inner_batch(U2dx,U2dx)+inner_batch(U2dy,U2dy)+\
                       inner_batch(V2dx,V2dx)+inner_batch(V2dy,V2dy))
    
    b = -inner_batch(3*U1-4*U_old+U_old_old,U2)-inner_batch(3*V1-4*V_old+V_old_old,V2)\
        -2*dt*(inner_batch(UgradU1,U1)+inner_batch(UgradU2,V1))
    
    c = -2*(inner_batch(U1-U_old,U1-U_old)+inner_batch(V1-V_old,V1-V_old))\
            +0.5*(inner_batch(U1-U_old_old,U1-U_old_old)+inner_batch(V1-V_old_old,V1-V_old_old))       
    
    # Scale by grid cell area and add the theta-term (theta assumed defined)
   
    a_new = a * hx * hy + 3*theta
    b_new = b * hx * hy
    c_new = c*hx*hy-theta*(4*q_old**2-q_old_old**2)
      
    # Solve the quadratic for q (taking the positive square root branch)
    delta = b_new**2-4*a_new*c_new
    q1_new = (-b_new+np.sqrt(delta))/(2*a_new)
    q2_new = (-b_new-np.sqrt(delta))/(2*a_new)
    
    q_new = np.where(np.abs(q1_new - 1) < np.abs(q2_new - 1), q1_new, q2_new)
    
#     if abs(q1_new-1)<abs(q2_new-1):
#         q_new = q1_new 
#     else:
#         q_new = q2_new
    
#     print(q_new)
    # Step 4: update U and V with the new correction.
    q_new_resh = q_new.reshape(1, 1, -1)
    U_new = U1 + q_new_resh * U2
    V_new = V1 + q_new_resh * V2
    egy = 0.5 * hx * hy * (inner_batch(U_new, U_new) + inner_batch(V_new, V_new))
#     qq = q_new

#     print("Error in the energy equation: {:.2e}".format(err_egy))

    # After the loop, update pressure:
    P_new = P1 + q_new * P2
    return U_new, V_new, P_new, q_new, egy