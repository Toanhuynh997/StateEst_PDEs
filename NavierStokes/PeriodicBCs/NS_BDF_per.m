function [U, V, P, P_store, qq, egy, perS, Xp, Yp] =...
    NS_BDF_per(xa,xb,ya,yb,T,Nx,Ny,Nt,mu,theta,opt)
% We solve the NS equations 
%       u_t = mu\Delta(u)+f-F(u)-grad(p)
%       div(u) = 0
% in 2D with periodic BCs, staggered grid 
% MAC scheme in space, BDF2 in time

hx = (xb-xa)/Nx; hy = (yb-ya)/Ny;
x = xa:hx:xb;    y = ya:hy:yb;
xmid = avg(x);   ymid = avg(y);
dt = T/Nt;       TT = 0:dt:T;

alpha = 1.5/dt; % alpha*u-mu*Delta(u)+nabla(p)=f
%--------------------------------------------------------------------------
% construct matrices A,B 
sU = Nx*Ny; % sV = sU; sP = sU; 
A0  = DiscreteLaplace(Nx,hx);
B0  = DiscreteLaplace(Ny,hy);
A_u = alpha*speye(sU)-mu*(kron(speye(Ny),A0)+kron(B0,speye(Nx)));
A_v = A_u;

A = blkdiag(A_u,A_v);
% perA = symamd(A); AA = chol(A(perA,perA)); AAt = AA';

A1 = DiscreteGrad(Nx,hx);         % P_x = A1*P
B1 = DiscreteGrad(Ny,hy);         % P_y = P*B1'
B = [kron(speye(Ny),A1') kron(B1',speye(Nx))];
B(1,:) = []; % assume P(1,1) = 0
Bt = B';
%--------------------------------------------------------------------------
dA = diag(A);
D  = spdiags(dA,0,size(A,1),size(A,2));
E  = D-A;
Di = spdiags(1./dA,0,size(A,1),size(A,2));
S  = B*Di*Bt; perS = symamd(S); SS = chol(S(perS,perS)); SSt = SS';
DiE = Di*E;
BDiE = B*DiE;
DiBt = Di*Bt;
%--------------------------------------------------------------------------
% Initilize U,V,q
[Yu,Xu] = meshgrid(ymid,x(1:end-1));
[Yv,Xv] = meshgrid(y(1:end-1),xmid);
[Yp,Xp] = meshgrid(ymid,xmid);
U = u_init(Xu,Yu,opt);
V = v_init(Xv,Yv,opt);
q = 1;
qq = zeros(Nt+1,1);
qq(1) = q;

egy = zeros(Nt+1,1);
egy(1) = 0.5*hx*hy*(inner(U,U)+inner(V,V));

U_old_old = U; V_old_old = V; q_old_old = q;
[U_old,V_old,P_old,q_old] = NS_bEuler_per(xa,xb,ya,yb,dt,Nx,Ny,1,mu,theta,opt);
egy(2) = 0.5*hx*hy*(inner(U_old,U_old)+inner(V_old,V_old));
qq(2) = q_old;

P_store = zeros(Nx*Ny,Nt);
P_store(:, 1) = reshape(P_old, Nx*Ny, 1);
tic
for k = 2:Nt
    % Step 1: find U1,V1,P1
    ff1 = f1(Xu,Yu,TT(k+1),mu,opt);
    ff2 = f2(Xv,Yv,TT(k+1),mu,opt);
    rhs1 = ff1+(4*U_old-U_old_old)/(2*dt);   
    rhs2 = ff2+(4*V_old-V_old_old)/(2*dt);
    rhs = [rhs1(:);rhs2(:)];
    [U1,V1,P1] = stokes_solver(rhs,DiE,BDiE,DiBt,Di,B,Bt,perS,SS,SSt,Nx,Ny,sU);

    % Step 2: find U2,V2,P2
    [UgradU1,UgradU2] = compute_UgradU(2*U_old-U_old_old,2*V_old-V_old_old,hx,hy);
    rhs = -[UgradU1(:);UgradU2(:)];
    [U2,V2,P2] = stokes_solver(rhs,DiE,BDiE,DiBt,Di,B,Bt,perS,SS,SSt,Nx,Ny,sU);
    
    % Step 3: find q
    U2dx = (-A1')*U2; U2dy = U2*(-B1);
    V2dx = (-A1')*V2; V2dy = V2*(-B1);

    a = 1.5*(inner(U2,U2)+inner(V2,V2)) ...
        +2*dt*mu*(inner(U2dx,U2dx)+inner(U2dy,U2dy)+inner(V2dx,V2dx)+inner(V2dy,V2dy));
    b = -inner(3*U1-4*U_old+U_old_old,U2) ...
        -inner(3*V1-4*V_old+V_old_old,V2) ...
        -2*dt*(inner(UgradU1,U1)+inner(UgradU2,V1));
    c = -2*(inner(U1-U_old,U1-U_old)+inner(V1-V_old,V1-V_old)) ...
        +0.5*(inner(U1-U_old_old,U1-U_old_old)+inner(V1-V_old_old,V1-V_old_old));
    a = a*hx*hy+3*theta; b = b*hx*hy; c = c*hx*hy-theta*(4*q_old^2-q_old_old^2);
    delta = b^2-4*a*c;
    if delta<0, fprintf('\nWarning: q at the %gth iteration is complex\n',k); break
    else, q1 = (-b+sqrt(delta))/(2*a);
          q2 = (-b-sqrt(delta))/(2*a);
          if abs(q1-1)<abs(q2-1), q = q1; else, q = q2; end
    end

    % Step 4: update U,V
    U = U1+q*U2; 
    V = V1+q*V2; 
    egy(k+1) = 0.5*hx*hy*(inner(U,U)+inner(V,V));
    qq(k+1) = q;

    % Prepare for next iteration
    U_old_old = U_old; U_old = U;
    V_old_old = V_old; V_old = V;
    q_old_old = q_old; q_old = q;

    P = P1+q*P2;

    P_store(:, k) = reshape(P, Nx*Ny, 1);
end
% P = P1+q*P2;
toc

save BDF2_Taylor_Green.mat egy qq xa xb ya yb T mu theta opt Nx Ny Nt

% Compute the error
[err_Linf_u,err_L2_u] = err(U,@u_exact,Xu,Yu,hx,hy,TT(k+1),mu,opt);
[err_Linf_v,err_L2_v] = err(V,@v_exact,Xv,Yv,hx,hy,TT(k+1),mu,opt);
[err_Linf_p,err_L2_p] = err(P,@p_exact,Xp,Yp,hx,hy,TT(k+1),mu,opt);

fprintf(1,'\n --- Running Results of BDF2 (option %d) ---\n',opt);
fprintf(1,'\n k = %g\n',k)
fprintf(1,'\n (Nx,Ny,Nt,mu,theta) = (%g,%g,%g,%.6g,%.6g)',Nx,Ny,Nt,mu,theta);
fprintf(1,'\n uLinf error = %.2e, uL2 error = %.2e',err_Linf_u,err_L2_u);
fprintf(1,'\n vLinf error = %.2e, vL2 error = %.2e',err_Linf_v,err_L2_v);
fprintf(1,'\n ULinf error = %.2e, UL2 error = %.2e',...
                    max(err_Linf_u,err_Linf_v),sqrt(err_L2_u^2+err_L2_v^2));
fprintf(1,'\n pLinf error = %.2e, pL2 error = %.2e',err_Linf_p,err_L2_p);
fprintf(1,'\n qLinf error = %.2e',abs(1-q));
fprintf(1,'\n For LaTeX: %.2e&%.2e&%.2e&%.2e&%.2e',...
                        err_L2_u,err_Linf_u,err_L2_p,err_Linf_p,abs(1-q));
fprintf(1,'\n *************************************************\n\n');

figure
plot(TT,qq,LineWidth=1)
legend('Evolution of q')

if     opt == 1, egy_ex = pi^2*exp(-4*mu*TT);
elseif opt == 2, egy_ex = 0.25*exp(-16*pi^2*mu*TT); end

figure
plot(TT,egy,LineWidth=2)
hold on
plot(TT,egy_ex,LineWidth=2)
legend('\boldmath$\mathcal{K}(u^n)$',...
    '\boldmath$\mathcal{K}(u)$','Interpreter','latex')
xlabel('\bf{Time}')
ylabel('\bf{Energy}')
set(gca, 'FontWeight', 'bold', 'linewidth',1.5,'fontsize',20)

%=======================================================================

function B = avg(A) % same as diff, but average
if size(A,1)==1, B = (A(2:end)+A(1:end-1))/2;
else,            B = (A(2:end,:)+A(1:end-1,:))/2; end
%---
function uu = u_exact(x,y,t,mu,opt)
switch opt
    case 1, uu = sin(x).*cos(y)*exp(-2*mu*t);
    case 2, uu = sin(2*pi*x).*cos(2*pi*y)*exp(-8*pi^2*mu*t);
end

function vv = v_exact(x,y,t,mu,opt)
switch opt
    case 1, vv = -cos(x).*sin(y)*exp(-2*mu*t);
    case 2, vv = -cos(2*pi*x).*sin(2*pi*y)*exp(-8*pi^2*mu*t);
end

function pp = p_exact(x,y,t,mu,opt)
switch opt
    case 1, pp = 0.25*(cos(2*x)+cos(2*y))*exp(-4*mu*t);
    case 2, pp = 0.25*(cos(4*pi*x)+cos(4*pi*y))*exp(-16*pi^2*mu*t);
end
%---
function u0 = u_init(x,y,opt)
switch opt
    case 1, u0 = sin(x).*cos(y);
    case 2, u0 = sin(2*pi*x).*cos(2*pi*y);
end

function v0 = v_init(x,y,opt)
switch opt
    case 1, v0 = -cos(x).*sin(y);
    case 2, v0 = -cos(2*pi*x).*sin(2*pi*y);
end
%---
function ff1 = f1(x,y,t,mu,opt)
switch opt
    case {1,2}, ff1 = 0*x+0*y+0*t+0*mu;
end

function ff2 = f2(x,y,t,mu,opt)
switch opt
    case {1,2}, ff2 = 0*x+0*y+0*t+0*mu;
end

function r = inner(U,V)
r = sum(sum(U.*V));

function [err_Linf,err_L2] = err(W,func,X,Y,hx,hy,t,mu,opt)
r = func(X,Y,t,mu,opt)-W;
err_Linf = max(max(abs(r)));
err_L2 = sqrt(hx*hy*sum(sum(r.^2)));

function [UgradU1,UgradU2] = compute_UgradU(U,V,hx,hy)
% U, V: periodic BCs
[Udx,Udy] = grad_velo(U,hx,hy);
Vmod = [V(end,:);V]; Vmod = [Vmod,Vmod(:,1)];
Vmod = avg(Vmod); Vmod = avg(Vmod')';
UgradU1 = U.*Udx + Vmod.*Udy;

[Vdx,Vdy] = grad_velo(V,hx,hy);
Umod = [U(:,end),U]; Umod = [Umod;Umod(1,:)];
Umod = avg(Umod); Umod = avg(Umod')';
UgradU2 = Umod.*Vdx + V.*Vdy;

function [Udx,Udy] = grad_velo(U,hx,hy)
Udx = [U(end,:);U;U(1,:)];
Udx = (Udx(3:end,:)-Udx(1:end-2,:))/(2*hx); % Udx~U
Udy = [U(:,end),U,U(:,1)];
Udy = (Udy(:,3:end)-Udy(:,1:end-2))/(2*hy); % Udy~U

function A = DiscreteGrad(N,h)
% size(A) = [N,N], periodic
A = spdiags(ones(N,1)*[-1 1],-1:0,N,N)/h;
A(1,end) = -1/h;

function A = DiscreteLaplace(N,h)
% periodic
A = spdiags(ones(N,1)*[1 -2 1],-1:1,N,N)/h^2;
A(1,end) = 1/h^2; A(end,1) = 1/h^2;

function [U,V,P] = stokes_solver(f,DiE,BDiE,DiBt,Di,B,Bt,perS,SS,SSt,Nx,Ny,sU)
% solve alpha*u-mu*Delta(u)+grad(p) = f
%                            div(v) = 0
%                                 u = 0 on the boundary
lhs = @(u) lhs_u(u,DiE,BDiE,DiBt,perS,SS,SSt);
rhs_1 = Di*f;
rhs_2 = B*rhs_1;
rhs_2(perS) = SS\(SSt\rhs_2(perS));
rhs = rhs_1-Di*Bt*rhs_2;
u = gmres(lhs,rhs,[],1e-8,1000,[],[],[]);
p = BDiE*u;
p(perS) = SS\(SSt\p(perS));
p = p+rhs_2;
%---
U = reshape(u(1:sU),Nx,Ny);     % periodic (Nx,Ny)
V = reshape(u(sU+1:end),Nx,Ny); % periodic (Nx,Ny)
p_ex = [0;p];
P = reshape(p_ex-mean(p_ex),Nx,Ny);

function lhs = lhs_u(u,DiE,BDiE,DiBt,perS,SS,SSt)
lhs = BDiE*u;
lhs(perS) = SS\(SSt\lhs(perS));
lhs = u-DiE*u+DiBt*lhs;