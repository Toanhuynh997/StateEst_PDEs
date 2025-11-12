% function [U,V,P,q, Xp, Yp, Xu, Yu, Xv, Yv, perS, qq, egy] = ....
%     NS_bEuler_ChannelFlow_NoCyl(U0, V0, xa,xb,ya,yb,T,Nx,Ny,Nt,mu,theta)
function [U_store,V_store,P_store,q, Xp, Yp, Xu, Yu, Xv, Yv, perS, qq, egy] = ....
    NS_bEuler_ChannelFlow_NoCyl_v3(U0, V0, xa,xb,ya,yb, dt, T,Nx,Ny,Nt,...
    mu,theta, opt)

    % We solve the NS equations 
    %       u_t = mu\Delta(u)+f-F(u)-grad(p)
    %       div(u) = 0
    % in 2D with u = 0 on the boundary, staggered grid 
    % MAC scheme in space, backward Euler in time
    
    hx = (xb-xa)/Nx; hy = (yb-ya)/Ny;
    x = xa:hx:xb;    y = ya:hy:yb;
    xmid = avg(x);   ymid = avg(y);
    % dt = T/Nt;       
    TT = 0:dt:T;
    
    alpha = 1/dt; % alpha*u-mu*Delta(u)+nabla(p)=f
    % opt_UgradU = 1; % 1:original, 2:MIT (not good)
    %--------------------------------------------------------------------------
    % construct matrices A,B 
    sU = Nx*Ny; 
    sV = Nx*(Ny-1); 
    sP = Nx*Ny; 

    %% Matrix A
    %%%%%% A_u
    A0  = DiscreteLaplace(Nx,hx);
    dd1 = ones(sU,1);
    dd0 = [-3*ones(Nx,1);-2*ones(sU-2*(Nx),1);-3*ones(Nx,1)];
    % TT_A = spdiags([dd1 dd0 dd1],[-Nx 0 Nx],sU,sU)/hy^2;
    A_u = alpha*speye(sU)-mu*(kron(speye(Ny),A0)...
        +spdiags([dd1 dd0 dd1],[-Nx 0 Nx],sU,sU)/hy^2);
    %%%%%% A_v
    Asup = spdiags(ones(Nx,1)*[1 -2 1],-1:1,Nx,Ny)/hx^2;
    Asup(1,end) = 1/hx^2; 
    Asup(end,1) = 1/hx^2;

    A_v = speye(Nx*(Ny-1))*alpha-mu*(kron(speye(Ny-1), Asup) ...
                     +spdiags(ones(sV,1)*[1 -2 1],[-Nx 0 Nx],sV,sV)/hy^2);
    % A_v = A_u;
    % d_1 = repmat([ones(Nx-1,1);0],Ny-1,1); d1 = [0;d_1(1:end-1)];
    % d0  = repmat([-3;-2*ones(Nx-2,1);-3],Ny-1,1);
    % A_v = speye(Nx*(Ny-1))*alpha-mu*(spdiags([d_1 d0 d1],-1:1,sV,sV)/hx^2 ...
    %     +spdiags(ones(sV,1)*[1 -2 1],[-Nx 0 Nx],sV,sV)/hy^2);
    
    A = blkdiag(A_u,A_v);
    % perA = symamd(A); 
    % AA = chol(A(perA,perA)); 
    % AAt = AA';
    
    A1 = DiscreteGrad(Nx,hx);         % P_x = A1*P
    B1 = (spdiags(ones(Nx,1)*[1 -1],-1:0,Nx,Ny-1)/hy)';    % P_y = P*B1'
    %% Matrix B
    % r1 = 1:sP; 
    % r1(Nx:Nx:end) = [];
    % r2 = 1:sP;
    % r2(1:Nx:end)  = [];
    B_u = kron(speye(Ny),A1');
    B_v = kron(B1',speye(Nx));
    B = [B_u B_v];
    B(1,:) = []; % assume P(1,1) = 0
    Bt = B';

    %--------------------------------------------------------------------------
    dA = diag(A);
    D  = spdiags(dA,0,size(A,1),size(A,2));
    E  = D-A;
    Di = spdiags(1./dA,0,size(A,1),size(A,2));
    S  = B*Di*Bt; 
    perS = symamd(S);
    SS = chol(S(perS,perS)); 
    SSt = SS';

    DiE = Di*E;
    BDiE = B*DiE;
    DiBt = Di*Bt;

    [Yu,Xu] = meshgrid(ymid,x(1:end-1));
    [Yv,Xv] = meshgrid(y(2:end-1),xmid);
    [Yp,Xp] = meshgrid(ymid,xmid);

    %% Initial conditions
    U = U0;
    V = V0;
    % U = zeros(Nx, Ny);
    % V = zeros(Nx, Ny-1);

    q = 1;
    qq = zeros(Nt+1,1);
    qq(1) = q;
    
    egy = zeros(Nt+1,1);
    egy(1) = 0.5*hx*hy*(inner(U,U)+inner(V,V));
    U_store = zeros(Nt+1, Nx*Ny);
    V_store = zeros(Nt+1, Nx*(Ny-1));
    P_store = zeros(Nt+1, Nx*Ny);

    U_store(1, :) = reshape(U0, 1, Nx*Ny);
    V_store(1, :) = reshape(V0, 1, Nx*(Ny-1));
    P_store(1, :) = zeros(1, sP);
    for k = 1:Nt
        k
        %% Step 1: find U1,V1,P1
        ff1 = f1(Xu,Yu,TT(k+1),mu, opt); 
        rhs1 = alpha*U+ff1; 
        ff2 = f2(Xv,Yv,TT(k+1),mu, opt); 
        rhs2 = alpha*V+ff2;  
        
        rhs = [rhs1(:);rhs2(:)];
        [U1, V1, P1] = stokes_solver(rhs,DiE,BDiE,DiBt,Di,B,Bt,perS,...
            SS,SSt,Nx,Ny,sU);

        %% Step 2: find U2,V2,P2

        [UgradU1,UgradU2] = compute_UgradU(U,V,hx,hy);
        rhs = -[UgradU1(:);UgradU2(:)];
        [U2, V2, P2] = stokes_solver(rhs,DiE,BDiE,DiBt,Di,B,Bt,perS,SS,...
            SSt,Nx,Ny,sU);

        %% Step 3: Find q
        % U_old = U;
        % V_old = V;
        % q_old = q;

        U2dx = (-A1')*U2;
        
        U2dy  = diff(U2')'/hy;       
        U2dy1 = 2*U2(:,1)/hy;        
        U2dy2 = -2*U2(:,end)/hy;     

        V2dx = (-A1')*V2;

        V_ext = zeros(size(V2)+[0 2]); 
        V_ext(:,2:end-1) = V2;
        V2dy = diff(V_ext')'/hy;

        a = 0.5*(inner(U2,U2)+inner(V2,V2)) ...
            +dt*mu*(inner(U2dx,U2dx)+inner(U2dy,U2dy)+...
            inner(V2dx,V2dx)+inner(V2dy,V2dy))+...
            +dt*mu*(inner(U2dy1,U2dy1)+inner(U2dy2,U2dy2))/2;

        b = inner(U-U1,U2)+inner(V-V1,V2)...
            -dt*(inner(UgradU1,U1)+inner(UgradU2,V1));

        c = -0.5*(inner(U1-U,U1-U)+inner(V1-V,V1-V));
        a = a*hx*hy+theta;  
        b = b*hx*hy; 
        c = c*hx*hy-q^2*theta;

        q = (-b+sqrt(b^2-4*a*c))/(2*a);

        % Step 4: update U,V
        U = U1+q*U2;
        V = V1+q*V2;
        P = P1+q*P2;
        egy(k+1) = 0.5*hx*hy*(inner(U,U)+inner(V,V));
        qq(k+1) = q;

        U_store(k+1, :) = reshape(U, 1, Nx*Ny);
        V_store(k+1, :) = reshape(V, 1, Nx*(Ny-1));
        P_store(k+1, :) = reshape(P, 1, sP);
    end
    % P = P1+q*P2;
end