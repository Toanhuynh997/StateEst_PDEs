function [U,V,P] = ...
    stokes_solver(f,DiE,BDiE,DiBt,Di,B,Bt,perS,SS,SSt,Nx,Ny,sU)
    % solve alpha*u-mu*Delta(u)+grad(p) = f
    %                            div(u) = 0
    %                                 u = 0 on the boundary
    lhs = @(u) lhs_u(u,DiE,BDiE,DiBt,perS,SS,SSt);
    rhs_1 = Di*f;
    rhs_2 = B*rhs_1;
    rhs_2(perS) = SS\(SSt\rhs_2(perS));
    rhs = rhs_1-Di*Bt*rhs_2;
    u = gmres(lhs,rhs,[],1e-9,3000,[],[],[]);
    p = BDiE*u;
    p(perS) = SS\(SSt\p(perS));
    p = p+rhs_2;
    %---
    U = reshape(u(1:sU),Nx,Ny);
    V = reshape(u(sU+1:end),Nx,Ny-1);
    p_ex = [0;p];
    P = reshape(p_ex-mean(p_ex),Nx,Ny);
end