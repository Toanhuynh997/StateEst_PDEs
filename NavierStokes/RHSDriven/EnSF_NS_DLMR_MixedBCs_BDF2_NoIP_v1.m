clear
load('RefSol_BDF2_Mixed_PeriDiri_60_4000_opt2_Re2000v1.mat', 'U_Ref',...
    'V_Ref', 'P_Ref', 'q_Ref')

%% Setting
xa = 0; xb = 1; 
ya = 0; yb = 1; 
T = 15; opt = 2; 

Nx = 60; 

% Nx = 32; 
Ny = Nx; 
theta = 100;
% NtEnSF = 500;
NtEnSF = 1000;

hx = (xb-xa)/Nx; 
hy = (yb-ya)/Ny;
x = xa:hx:xb; 
y = ya:hy:yb;
xmid = avg(x);
ymid = avg(y);
dtEnSF = T/NtEnSF;    
TTEnSF = 0:dtEnSF:T;

sU = Nx*Ny; 
sV = Nx*(Ny-1); 
sP = Nx*Ny; 
mu_big = 1/2000;
alphaBE = 1/dtEnSF;

%% Matrix A for BE
%%%%%% A_u
A0  = DiscreteLaplace(Nx,hx);
dd1 = ones(sU,1);
dd0 = [-3*ones(Nx,1);-2*ones(sU-2*(Nx),1);-3*ones(Nx,1)];

A_u = alphaBE*speye(sU)-mu_big*(kron(speye(Ny),A0)...
    +spdiags([dd1 dd0 dd1],[-Nx 0 Nx],sU,sU)/hy^2);
%%%%%% A_v
Asup = spdiags(ones(Nx,1)*[1 -2 1],-1:1,Nx,Ny)/hx^2;
Asup(1,end) = 1/hx^2; 
Asup(end,1) = 1/hx^2;

A_v = speye(Nx*(Ny-1))*alphaBE-mu_big*(kron(speye(Ny-1), Asup) ...
                 +spdiags(ones(sV,1)*[1 -2 1],[-Nx 0 Nx],sV,sV)/hy^2);

A = blkdiag(A_u,A_v);

A1 = DiscreteGrad(Nx,hx);         % P_x = A1*P
B1 = (spdiags(ones(Nx,1)*[1 -1],-1:0,Nx,Ny-1)/hy)';    % P_y = P*B1'

B_u = kron(speye(Ny),A1');
B_v = kron(B1',speye(Nx));
B = [B_u B_v];
B(1,:) = []; % assume P(1,1) = 0
Bt = B';
dA = diag(A);
D  = spdiags(dA,0,size(A,1),size(A,2));
E  = D-A;
Di = spdiags(1./dA,0,size(A,1),size(A,2));
S  = B*Di*Bt; 
perS_EnSF = symamd(S);
SS = chol(S(perS_EnSF,perS_EnSF)); 
SSt = SS';

DiE = Di*E;
BDiE = B*DiE;
DiBt = Di*Bt;

[Yu,Xu] = meshgrid(ymid,x(1:end-1));
[Yv,Xv] = meshgrid(y(2:end-1),xmid);
[Yp,Xp] = meshgrid(ymid,xmid);

U0 = zeros(Nx, Ny);
V0 = zeros(Nx, Ny-1);

[mU0, nU0] = size(U0);
[mV0, nV0] = size(V0);

[mP0, nP0] = size(Xp);

Size_U = mU0*nU0;
Size_V = mV0*nV0;
Size_P = mP0*nP0;


%% 100%
idxU_obs = randperm(Size_U, floor(1*Size_U));  % shape => 1 x (1*Size_U)
idxV_obs = randperm(Size_V, floor(1*Size_V))+Size_U;
idxP_obs = randperm(Size_P, floor(1*Size_P))+Size_U+Size_V;
idx_q = Size_U+Size_V+Size_P+1;

% num_indices = numel(idxU_obs) + numel(idxV_obs) + numel(idxP_obs)+1;
num_indices = numel(idxU_obs) + numel(idxV_obs) + numel(idxP_obs);
idx_obs = sort([idxU_obs, idxV_obs, idxP_obs]);

%%% Model Uncertainties
SDE_Sigma_U = 0.01;
SDE_Sigma_V = 0.01;
SDE_Sigma_P = 0.0005;
SDE_Sigma_q = 0.0000001;

Nt = 2000;
t0 = 0;
filtering_steps = NtEnSF;

% timeTrue : shape (Nt+1)
timeTrue = linspace(0, 1, Nt+1);

% tEnSF : shape (filtering_steps+1)
tEnSF = linspace(0, 1, filtering_steps+1);
indices_time = zeros(size(tEnSF));
for i = 1:length(tEnSF)
    % find the first index j in timeTrue s.t. timeTrue(j) >= tEnSF(i)
    idx = find(timeTrue >= tEnSF(i), 1, 'first');
    if isempty(idx)
        % if tEnSF(i) is beyond the max of timeTrue
        idx = length(timeTrue);
    end
    indices_time(i) = idx;
end


state_ref = [U_Ref, V_Ref, P_Ref];
% state_ref = [U_Ref, V_Ref, P_Ref, q_Ref];
state_timeextract = state_ref(indices_time, :);
state_EnSF = state_timeextract(:, idx_obs);

obs_sigma = 0.1;
eps_alpha = 0.05;

% ensemble size
ensemble_size = 60;
ensemble_true = 1;

% forward Euler steps
euler_steps = 400;
U0_state = 2*randn(ensemble_size, Size_U);
V0_state = 2*randn(ensemble_size, Size_V);
P0_state = 2*randn(ensemble_size, Size_P);

UV_state = [U0_state, V0_state, P0_state];

% n_dim = Size_U+Size_V+Size_P+1;
n_dim = Size_U+Size_V+Size_P;
rmse_all = zeros(filtering_steps);   % e.g. can store root-mean-square errors over time
obs_save = [];   % maybe store observations?

est_save = zeros(filtering_steps+1, n_dim);

est_save(1, :) = est_save(1, :) + mean(UV_state, 1);

q = 1;

q_batch = q * ones(ensemble_size, 1);

%% Running EnSF

%%% First step: using BDF1 to find U1_ens
U_stack = (U0_state');
V_stack = (V0_state');

U_stack = reshape(U_stack, [mU0, nU0, ensemble_size]);
V_stack = reshape(V_stack, [mV0, nV0, ensemble_size]);


ff1 = repmat(f1(Xu,Yu,TTEnSF(2),mu_big, 2), [1, 1, ensemble_size]); 
ff2 = repmat(f2(Xv,Yv,TTEnSF(2),mu_big, 2), [1, 1, ensemble_size]);

ff1 = ff1+0.001*randn(size(Xu, 1), size(Xu, 2), ensemble_size);
ff2 = ff2+0.001*randn(size(Xv, 1), size(Xv, 2), ensemble_size);

[U_new, V_new, P_new, q_new] = ...
    NS_1st_BE_1step_vectorize(hx, hy, dtEnSF, U_stack, V_stack,...
    q_batch, mu_big, theta, DiE, BDiE, DiBt, Di,B, Bt, perS_EnSF,...
    SS, SSt, Nx, Ny, sU, alphaBE, A1, ff1, ff2);

U_new_reshape = reshape(U_new, [mU0*nU0, ensemble_size]); 
V_new_reshape = reshape(V_new, [mV0*nV0, ensemble_size]);
P_new_reshape = reshape(P_new, [mP0*nP0, ensemble_size]);

x_state = [U_new_reshape; V_new_reshape; P_new_reshape];

noiseU = sqrt(dtEnSF)*SDE_Sigma_U*randn(size(U_new_reshape));
noiseV = sqrt(dtEnSF)*SDE_Sigma_V*randn(size(V_new_reshape));
noiseP = sqrt(dtEnSF)*SDE_Sigma_P*randn(size(P_new_reshape));

noise = [noiseU; noiseV; noiseP];

x_state = x_state+noise;
x_state = x_state';

x0_EnSF = x_state(:, idx_obs);

%%% Using EnSF for U1_ens
state_scale = state_EnSF(2, :);

Vel_state = state_scale(1, 1:(numel(idxU_obs)+numel(idxV_obs)));
Pres_state = ...
    state_scale(1, numel(idxU_obs)+numel(idxV_obs)+(1:numel(idxP_obs)));

mask_pres1 = ( (-1e-1 <= Pres_state) & (Pres_state < -1e-2) ) |...
        ( (1e-2 <= Pres_state) & (Pres_state < 1e-1) );

mask_pres2 = ( (-1e-2 <= Pres_state) & (Pres_state < -1e-3) ) |...
    ( (1e-3 <= Pres_state) & (Pres_state < 1e-2) );

mask_pres3 = ( (-1e-3 <= Pres_state) & (Pres_state < -1e-4) ) |...
    ( (1e-4 <= Pres_state) & (Pres_state < 1e-3) );

mask_pres4 = ( (-1e-4 <= Pres_state) & (Pres_state < -1e-5) ) |...
    ( (1e-5 <= Pres_state) & (Pres_state < 1e-4) );

mask_pres5 = ( (-1e-5 <= Pres_state) & (Pres_state < -1e-6) ) |...
    ( (1e-6 <= Pres_state) & (Pres_state < 1e-5) );

mask_pres6 = ( (-1e-6 <= Pres_state) & (Pres_state < -1e-7) ) |...
    ( (1e-7 <= Pres_state) & (Pres_state < 1e-6) );

mask_pres7 = ( (-1e-7 <= Pres_state) & (Pres_state < -1e-8) ) |...
    ( (1e-8 <= Pres_state) & (Pres_state < 1e-7) );

mask_pres8 = ( (-1e-8 <= Pres_state) & (Pres_state < -1e-9) ) |...
    ( (1e-9 <= Pres_state) & (Pres_state < 1e-8) );

mask_pres9 = ( (-1e-9 <= Pres_state) & (Pres_state < -1e-10) ) |...
    ( (1e-10 <= Pres_state) & (Pres_state < 1e-9) );

mask_pres10 = ( (-1e-10 <= Pres_state) & (Pres_state < -1e-11) ) |...
    ( (1e-11 <= Pres_state) & (Pres_state < 1e-10) );

mask_pres11 = ( (-1e-11 <= Pres_state) & (Pres_state < -1e-12) ) |...
    ( (1e-12 <= Pres_state) & (Pres_state < 1e-11) );

mask_pres12 = ( (-1e-12 <= Pres_state) & (Pres_state < -1e-13) ) |...
    ( (1e-13 <= Pres_state) & (Pres_state < 1e-12) );

mask_pres13 = ( (-1e-13 <= Pres_state) & (Pres_state < -1e-14) ) |...
    ( (1e-14 <= Pres_state) & (Pres_state < 1e-13) );

mask_pres14 = ( (-1e-14 <= Pres_state) & (Pres_state < -1e-15) ) |...
    ( (1e-15 <= Pres_state) & (Pres_state < 1e-14) );

mask_pres15 = ( (-1e-15 <= Pres_state) & (Pres_state < -1e-16) ) |...
    ( (1e-16 <= Pres_state) & (Pres_state < 1e-15) );
mask_pres16 = ( (-1e-16 <= Pres_state) & (Pres_state < -1e-17) ) |...
    ( (1e-17 <= Pres_state) & (Pres_state < 1e-16) );
mask_pres17 = ( (-1e-17 <= Pres_state) & (Pres_state < 0) ) |...
    ( (0 <= Pres_state) & (Pres_state < 1e-17) );

indob_presscale1 = find(mask_pres1);
indob_presscale2 = find(mask_pres2);
indob_presscale3 = find(mask_pres3);
indob_presscale4 = find(mask_pres4);
indob_presscale5 = find(mask_pres5);
indob_presscale6 = find(mask_pres6);
indob_presscale7 = find(mask_pres7);
indob_presscale8 = find(mask_pres8);
indob_presscale9 = find(mask_pres9);
indob_presscale10 = find(mask_pres10);
indob_presscale11 = find(mask_pres11);
indob_presscale12 = find(mask_pres12);
indob_presscale13 = find(mask_pres13);
indob_presscale14 = find(mask_pres14);
indob_presscale15 = find(mask_pres15);
indob_presscale16 = find(mask_pres16);
indob_presscale17 = find(mask_pres17);

Pres_state(:, indob_presscale1) = Pres_state(:, indob_presscale1)*1e1;
Pres_state(:, indob_presscale2) = Pres_state(:, indob_presscale2)*1e2;
Pres_state(:, indob_presscale3) = Pres_state(:, indob_presscale3)*1e3;
Pres_state(:, indob_presscale4) = Pres_state(:, indob_presscale4)*1e4;
Pres_state(:, indob_presscale5) = Pres_state(:, indob_presscale5)*1e5;
Pres_state(:, indob_presscale6) = Pres_state(:, indob_presscale6)*1e6;
Pres_state(:, indob_presscale7) = Pres_state(:, indob_presscale7)*1e7;
Pres_state(:, indob_presscale8) = Pres_state(:, indob_presscale8)*1e8;
Pres_state(:, indob_presscale9) = Pres_state(:, indob_presscale9)*1e9;
Pres_state(:, indob_presscale10) = Pres_state(:, indob_presscale10)*1e10;
Pres_state(:, indob_presscale11) = Pres_state(:, indob_presscale11)*1e11;
Pres_state(:, indob_presscale12) = Pres_state(:, indob_presscale12)*1e12;
Pres_state(:, indob_presscale13) = Pres_state(:, indob_presscale13)*1e13;
Pres_state(:, indob_presscale14) = Pres_state(:, indob_presscale14)*1e14;
Pres_state(:, indob_presscale15) = Pres_state(:, indob_presscale15)*1e15;
Pres_state(:, indob_presscale16) = Pres_state(:, indob_presscale16)*1e16;
Pres_state(:, indob_presscale17) = Pres_state(:, indob_presscale17)*1e17;

state_scale(1, 1:(numel(idxU_obs)+numel(idxV_obs)+numel(idxP_obs))) =...
    [Vel_state, Pres_state];

obs = atan(state_scale);
obs = obs+ randn(size(state_EnSF(2, :))) * obs_sigma;
for l =1:5
    Vel_EnSF = x0_EnSF(:, 1:(numel(idxU_obs)+numel(idxV_obs)));
    Pres_EnSF = ...
        x0_EnSF(:, numel(idxU_obs)+numel(idxV_obs)+(1:numel(idxP_obs)));

    mask_pres1_EnSF = ( (-1e-1 <= Pres_EnSF) & (Pres_EnSF < -1e-2) ) |...
        ( (1e-2 <= Pres_EnSF) & (Pres_EnSF < 1e-1) );

    mask_pres2_EnSF = ( (-1e-2 <= Pres_EnSF) & (Pres_EnSF < -1e-3) ) |...
        ( (1e-3 <= Pres_EnSF) & (Pres_EnSF < 1e-2) );

    mask_pres3_EnSF = ( (-1e-3 <= Pres_EnSF) & (Pres_EnSF < -1e-4) ) |...
        ( (1e-4 <= Pres_EnSF) & (Pres_EnSF < 1e-3) );

    mask_pres4_EnSF = ( (-1e-4 <= Pres_EnSF) & (Pres_EnSF < -1e-5) ) |...
        ( (1e-5 <= Pres_EnSF) & (Pres_EnSF < 1e-4) );

    mask_pres5_EnSF = ( (-1e-5 <= Pres_EnSF) & (Pres_EnSF < -1e-6) ) |...
        ( (1e-6 <= Pres_EnSF) & (Pres_EnSF < 1e-5) );

    mask_pres6_EnSF = ( (-1e-6 <= Pres_EnSF) & (Pres_EnSF < -1e-7) ) |...
        ( (1e-7 <= Pres_EnSF) & (Pres_EnSF < 1e-6) );

    mask_pres7_EnSF = ( (-1e-7 <= Pres_EnSF) & (Pres_EnSF < -1e-8) ) |...
        ( (1e-8 <= Pres_EnSF) & (Pres_EnSF < 1e-7) );
    mask_pres8_EnSF = ( (-1e-8 <= Pres_EnSF) & (Pres_EnSF < -1e-9) ) |...
        ( (1e-9 <= Pres_EnSF) & (Pres_EnSF < 1e-8) );

    mask_pres9_EnSF = ( (-1e-9 <= Pres_EnSF) & (Pres_EnSF < -1e-10) ) |...
        ( (1e-10 <= Pres_EnSF) & (Pres_EnSF < 1e-9) );
    mask_pres10_EnSF = ( (-1e-10 <= Pres_EnSF) & (Pres_EnSF < -1e-11) ) |...
        ( (1e-11 <= Pres_EnSF) & (Pres_EnSF < 1e-10) );
    mask_pres11_EnSF = ( (-1e-11 <= Pres_EnSF) & (Pres_EnSF < -1e-12) ) |...
            ( (1e-12 <= Pres_EnSF) & (Pres_EnSF < 1e-11) );
    mask_pres12_EnSF = ( (-1e-12 <= Pres_EnSF) & (Pres_EnSF < -1e-13) ) |...
            ( (1e-13 <= Pres_EnSF) & (Pres_EnSF < 1e-12) );
    mask_pres13_EnSF = ( (-1e-13 <= Pres_EnSF) & (Pres_EnSF < -1e-14) ) |...
            ( (1e-14 <= Pres_EnSF) & (Pres_EnSF < 1e-13) );
    mask_pres14_EnSF = ( (-1e-14 <= Pres_EnSF) & (Pres_EnSF < -1e-15) ) |...
            ( (1e-15 <= Pres_EnSF) & (Pres_EnSF < 1e-14) );
    mask_pres15_EnSF = ( (-1e-15 <= Pres_EnSF) & (Pres_EnSF < -1e-16) ) |...
            ( (1e-16 <= Pres_EnSF) & (Pres_EnSF < 1e-15) );
    mask_pres16_EnSF = ( (-1e-16 <= Pres_EnSF) & (Pres_EnSF < -1e-17) ) |...
            ( (1e-17 <= Pres_EnSF) & (Pres_EnSF < 1e-16) );
    mask_pres17_EnSF = ( (-1e-17 <= Pres_EnSF) & (Pres_EnSF < 0) ) |...
            ( (0 <= Pres_EnSF) & (Pres_EnSF < 1e-17) );

    [row_pres1, col_pres1] = find(mask_pres1_EnSF);
    [row_pres2, col_pres2] = find(mask_pres2_EnSF);
    [row_pres3, col_pres3] = find(mask_pres3_EnSF);
    [row_pres4, col_pres4] = find(mask_pres4_EnSF);
    [row_pres5, col_pres5] = find(mask_pres5_EnSF);
    [row_pres6, col_pres6] = find(mask_pres6_EnSF);
    [row_pres7, col_pres7] = find(mask_pres7_EnSF);
    [row_pres8, col_pres8] = find(mask_pres8_EnSF);
    [row_pres9, col_pres9] = find(mask_pres9_EnSF);
    [row_pres10, col_pres10] = find(mask_pres10_EnSF);
    [row_pres11, col_pres11] = find(mask_pres11_EnSF);
    [row_pres12, col_pres12] = find(mask_pres12_EnSF);
    [row_pres13, col_pres13] = find(mask_pres13_EnSF);
    [row_pres14, col_pres14] = find(mask_pres14_EnSF);
    [row_pres15, col_pres15] = find(mask_pres15_EnSF);
    [row_pres16, col_pres16] = find(mask_pres16_EnSF);
    [row_pres17, col_pres17] = find(mask_pres17_EnSF);

    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres1, col_pres1)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres1, col_pres1))*1e1;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres2, col_pres2)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres2, col_pres2))*1e2;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres3, col_pres3)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres3, col_pres3))*1e3;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres4, col_pres4)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres4, col_pres4))*1e4;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres5, col_pres5)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres5, col_pres5))*1e5;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres6, col_pres6)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres6, col_pres6))*1e6;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres7, col_pres7)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres7, col_pres7))*1e7;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres8, col_pres8)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres8, col_pres8))*1e8;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres9, col_pres9)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres9, col_pres9))*1e9;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres10, col_pres10)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres10, col_pres10))*1e10;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres11, col_pres11)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres11, col_pres11))*1e11;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres12, col_pres12)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres12, col_pres12))*1e12;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres13, col_pres13)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres13, col_pres13))*1e13;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres14, col_pres14)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres14, col_pres14))*1e14;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres15, col_pres15)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres15, col_pres15))*1e15;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres16, col_pres16)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres16, col_pres16))*1e16;
    Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres17, col_pres17)) ...
        = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres17, col_pres17))*1e17;

    x0_EnSF(:, 1:(numel(idxU_obs)+numel(idxV_obs)+numel(idxP_obs))) = ...
        [Vel_EnSF, Pres_EnSF];
    
    sln_bar = reverse_SDE(x0_EnSF, obs, euler_steps, 1, 1,...
        false, eps_alpha, obs_sigma);
    
    Vel_EnSF = sln_bar(:, 1:(numel(idxU_obs)+numel(idxV_obs)));
    Pres_EnSF = ...
        sln_bar(:, (numel(idxU_obs)+numel(idxV_obs))+(1:numel(idxP_obs)));

    Pres_EnSF(:, indob_presscale1) = Pres_EnSF(:, indob_presscale1)/1e1;
    Pres_EnSF(:, indob_presscale2) = Pres_EnSF(:, indob_presscale2)/1e2;
    Pres_EnSF(:, indob_presscale3) = Pres_EnSF(:, indob_presscale3)/1e3;
    Pres_EnSF(:, indob_presscale4) = Pres_EnSF(:, indob_presscale4)/1e4;
    Pres_EnSF(:, indob_presscale5) = Pres_EnSF(:, indob_presscale5)/1e5;
    Pres_EnSF(:, indob_presscale6) = Pres_EnSF(:, indob_presscale6)/1e6;
    Pres_EnSF(:, indob_presscale7) = Pres_EnSF(:, indob_presscale7)/1e7;
    Pres_EnSF(:, indob_presscale8) = Pres_EnSF(:, indob_presscale8)/1e8;
    Pres_EnSF(:, indob_presscale9) = Pres_EnSF(:, indob_presscale9)/1e9;
    Pres_EnSF(:, indob_presscale10) = ...
        Pres_EnSF(:, indob_presscale10)/1e10;
    Pres_EnSF(:, indob_presscale11) = ...
        Pres_EnSF(:, indob_presscale11)/1e11;
    Pres_EnSF(:, indob_presscale12) = ...
        Pres_EnSF(:, indob_presscale12)/1e12;
    Pres_EnSF(:, indob_presscale13) = ...
        Pres_EnSF(:, indob_presscale13)/1e13;
    Pres_EnSF(:, indob_presscale14) = ...
        Pres_EnSF(:, indob_presscale14)/1e14;
    Pres_EnSF(:, indob_presscale15) = ...
        Pres_EnSF(:, indob_presscale15)/1e15;
    Pres_EnSF(:, indob_presscale16) = ...
        Pres_EnSF(:, indob_presscale16)/1e16;
    Pres_EnSF(:, indob_presscale17) = ...
        Pres_EnSF(:, indob_presscale17)/1e17;

    x0_EnSF(:, 1:(numel(idxU_obs)+numel(idxV_obs)+numel(idxP_obs))) = ...
        [Vel_EnSF, Pres_EnSF];
end
x_state(:, idx_obs) = x0_EnSF;

% cols = (Size_U + Size_V) + (1:Size_P); 
% if i < 400
%     x_state(:, 1:Size_U) = min( max( x_state(:, 1:Size_U), -0.25), 0.25);
%     x_state(:, Size_U+(1:Size_V)) =...
%         min( max( x_state(:, Size_U+(1:Size_V)), -0.5), 0.5);
% else
%     x_state(:, 1:Size_U) = min( max( x_state(:, 1:Size_U), -0.35), 0.35);
%     x_state(:, Size_U+(1:Size_V)) =...
%         min( max( x_state(:, Size_U+(1:Size_V)), -0.33), 0.33);
% end

Unew = x_state(:, 1:Size_U);
Vnew = x_state(:, Size_U+(1:Size_V));

Unew = reshape(Unew', [mU0, nU0, ensemble_size]);
Vnew = reshape(Vnew', [mV0, nV0, ensemble_size]);

Uold = reshape(U0_state', [mU0, nU0, ensemble_size]);
Vold = reshape(V0_state', [mV0, nV0, ensemble_size]);

q_update = Update_q_EnSF(q_batch, Unew, Vnew, Uold, Vold,...
        hx, hy, dtEnSF, theta);


U00_state = U0_state;
V00_state = V0_state;

U0_state = x_state(:, 1:Size_U);
V0_state = x_state(:, Size_U+(1:Size_V));

est_save(2, :) = est_save(2, :) + mean(x_state, 1);

q0_batch = q_update;
q00_batch = q_batch;

%%% Using BDF2 from here
%% 
alphaBDF2 = 1.5/dtEnSF;
%% Matrix A for BDF2
%%%%%% A_u
A0  = DiscreteLaplace(Nx,hx);
dd1 = ones(sU,1);
dd0 = [-3*ones(Nx,1);-2*ones(sU-2*(Nx),1);-3*ones(Nx,1)];
% TT_A = spdiags([dd1 dd0 dd1],[-Nx 0 Nx],sU,sU)/hy^2;
A_u = alphaBDF2*speye(sU)-mu_big*(kron(speye(Ny),A0)...
    +spdiags([dd1 dd0 dd1],[-Nx 0 Nx],sU,sU)/hy^2);
%%%%%% A_v
Asup = spdiags(ones(Nx,1)*[1 -2 1],-1:1,Nx,Ny)/hx^2;
Asup(1,end) = 1/hx^2; 
Asup(end,1) = 1/hx^2;

A_v = speye(Nx*(Ny-1))*alphaBDF2-mu_big*(kron(speye(Ny-1), Asup) ...
                 +spdiags(ones(sV,1)*[1 -2 1],[-Nx 0 Nx],sV,sV)/hy^2);

A = blkdiag(A_u,A_v);

%% For BDF2
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

for i = 2:Nt
    Uold_stack = (U0_state');
    Vold_stack = (V0_state');
    
    Uold_old_stack = (U00_state');
    Vold_old_stack = (V00_state');

    Uold_stack = reshape(Uold_stack, [mU0, nU0, ensemble_size]);
    Vold_stack = reshape(Vold_stack, [mV0, nV0, ensemble_size]);
    
    Uold_old_stack = reshape(Uold_old_stack, [mU0, nU0, ensemble_size]);
    Vold_old_stack = reshape(Vold_old_stack, [mV0, nV0, ensemble_size]);

    ff1 = repmat(f1(Xu,Yu,TTEnSF(i+1),mu_big, 2), [1, 1, ensemble_size]); 
    ff2 = repmat(f2(Xv,Yv,TTEnSF(i+1),mu_big, 2), [1, 1, ensemble_size]);
    
    ff1 = ff1+0.001*randn(size(Xu, 1), size(Xu, 2), ensemble_size);
    ff2 = ff2+0.001*randn(size(Xv, 1), size(Xv, 2), ensemble_size);
    
    [U_new, V_new, P_new, q_new] = ...
        NS_BDF2_1step_vectorize(hx, hy, dtEnSF, Uold_stack, Uold_old_stack,...
        Vold_stack, Vold_old_stack, q0_batch, q00_batch, mu_big, theta, DiE,...
        BDiE, DiBt, Di,B, Bt, perS_EnSF, SS, SSt, Nx, Ny, sU,alphaBDF2, A1,...
        ff1, ff2);
    
    U_new_reshape = reshape(U_new, [mU0*nU0, ensemble_size]); 
    V_new_reshape = reshape(V_new, [mV0*nV0, ensemble_size]);
    P_new_reshape = reshape(P_new, [mP0*nP0, ensemble_size]);
    
    x_state = [U_new_reshape; V_new_reshape; P_new_reshape];
    
    noiseU = sqrt(dtEnSF)*SDE_Sigma_U*randn(size(U_new_reshape));
    noiseV = sqrt(dtEnSF)*SDE_Sigma_V*randn(size(V_new_reshape));
    noiseP = sqrt(dtEnSF)*SDE_Sigma_P*randn(size(P_new_reshape));
    
    noise = [noiseU; noiseV; noiseP];
    
    x_state = x_state+noise;
    x_state = x_state';
    
    x0_EnSF = x_state(:, idx_obs);
    
    %%% Using EnSF for U1_ens
    state_scale = state_EnSF(2, :);
    
    Vel_state = state_scale(1, 1:(numel(idxU_obs)+numel(idxV_obs)));
    Pres_state = ...
        state_scale(1, numel(idxU_obs)+numel(idxV_obs)+(1:numel(idxP_obs)));
    
    mask_pres1 = ( (-1e-1 <= Pres_state) & (Pres_state < -1e-2) ) |...
            ( (1e-2 <= Pres_state) & (Pres_state < 1e-1) );
    
    mask_pres2 = ( (-1e-2 <= Pres_state) & (Pres_state < -1e-3) ) |...
        ( (1e-3 <= Pres_state) & (Pres_state < 1e-2) );
    
    mask_pres3 = ( (-1e-3 <= Pres_state) & (Pres_state < -1e-4) ) |...
        ( (1e-4 <= Pres_state) & (Pres_state < 1e-3) );
    
    mask_pres4 = ( (-1e-4 <= Pres_state) & (Pres_state < -1e-5) ) |...
        ( (1e-5 <= Pres_state) & (Pres_state < 1e-4) );
    
    mask_pres5 = ( (-1e-5 <= Pres_state) & (Pres_state < -1e-6) ) |...
        ( (1e-6 <= Pres_state) & (Pres_state < 1e-5) );
    
    mask_pres6 = ( (-1e-6 <= Pres_state) & (Pres_state < -1e-7) ) |...
        ( (1e-7 <= Pres_state) & (Pres_state < 1e-6) );
    
    mask_pres7 = ( (-1e-7 <= Pres_state) & (Pres_state < -1e-8) ) |...
        ( (1e-8 <= Pres_state) & (Pres_state < 1e-7) );
    
    mask_pres8 = ( (-1e-8 <= Pres_state) & (Pres_state < -1e-9) ) |...
        ( (1e-9 <= Pres_state) & (Pres_state < 1e-8) );
    
    mask_pres9 = ( (-1e-9 <= Pres_state) & (Pres_state < -1e-10) ) |...
        ( (1e-10 <= Pres_state) & (Pres_state < 1e-9) );
    
    mask_pres10 = ( (-1e-10 <= Pres_state) & (Pres_state < -1e-11) ) |...
        ( (1e-11 <= Pres_state) & (Pres_state < 1e-10) );
    
    mask_pres11 = ( (-1e-11 <= Pres_state) & (Pres_state < -1e-12) ) |...
        ( (1e-12 <= Pres_state) & (Pres_state < 1e-11) );
    
    mask_pres12 = ( (-1e-12 <= Pres_state) & (Pres_state < -1e-13) ) |...
        ( (1e-13 <= Pres_state) & (Pres_state < 1e-12) );
    
    mask_pres13 = ( (-1e-13 <= Pres_state) & (Pres_state < -1e-14) ) |...
        ( (1e-14 <= Pres_state) & (Pres_state < 1e-13) );
    
    mask_pres14 = ( (-1e-14 <= Pres_state) & (Pres_state < -1e-15) ) |...
        ( (1e-15 <= Pres_state) & (Pres_state < 1e-14) );
    
    mask_pres15 = ( (-1e-15 <= Pres_state) & (Pres_state < -1e-16) ) |...
        ( (1e-16 <= Pres_state) & (Pres_state < 1e-15) );
    mask_pres16 = ( (-1e-16 <= Pres_state) & (Pres_state < -1e-17) ) |...
        ( (1e-17 <= Pres_state) & (Pres_state < 1e-16) );
    mask_pres17 = ( (-1e-17 <= Pres_state) & (Pres_state < 0) ) |...
        ( (0 <= Pres_state) & (Pres_state < 1e-17) );
    
    indob_presscale1 = find(mask_pres1);
    indob_presscale2 = find(mask_pres2);
    indob_presscale3 = find(mask_pres3);
    indob_presscale4 = find(mask_pres4);
    indob_presscale5 = find(mask_pres5);
    indob_presscale6 = find(mask_pres6);
    indob_presscale7 = find(mask_pres7);
    indob_presscale8 = find(mask_pres8);
    indob_presscale9 = find(mask_pres9);
    indob_presscale10 = find(mask_pres10);
    indob_presscale11 = find(mask_pres11);
    indob_presscale12 = find(mask_pres12);
    indob_presscale13 = find(mask_pres13);
    indob_presscale14 = find(mask_pres14);
    indob_presscale15 = find(mask_pres15);
    indob_presscale16 = find(mask_pres16);
    indob_presscale17 = find(mask_pres17);
    
    Pres_state(:, indob_presscale1) = Pres_state(:, indob_presscale1)*1e1;
    Pres_state(:, indob_presscale2) = Pres_state(:, indob_presscale2)*1e2;
    Pres_state(:, indob_presscale3) = Pres_state(:, indob_presscale3)*1e3;
    Pres_state(:, indob_presscale4) = Pres_state(:, indob_presscale4)*1e4;
    Pres_state(:, indob_presscale5) = Pres_state(:, indob_presscale5)*1e5;
    Pres_state(:, indob_presscale6) = Pres_state(:, indob_presscale6)*1e6;
    Pres_state(:, indob_presscale7) = Pres_state(:, indob_presscale7)*1e7;
    Pres_state(:, indob_presscale8) = Pres_state(:, indob_presscale8)*1e8;
    Pres_state(:, indob_presscale9) = Pres_state(:, indob_presscale9)*1e9;
    Pres_state(:, indob_presscale10) = Pres_state(:, indob_presscale10)*1e10;
    Pres_state(:, indob_presscale11) = Pres_state(:, indob_presscale11)*1e11;
    Pres_state(:, indob_presscale12) = Pres_state(:, indob_presscale12)*1e12;
    Pres_state(:, indob_presscale13) = Pres_state(:, indob_presscale13)*1e13;
    Pres_state(:, indob_presscale14) = Pres_state(:, indob_presscale14)*1e14;
    Pres_state(:, indob_presscale15) = Pres_state(:, indob_presscale15)*1e15;
    Pres_state(:, indob_presscale16) = Pres_state(:, indob_presscale16)*1e16;
    Pres_state(:, indob_presscale17) = Pres_state(:, indob_presscale17)*1e17;
    
    state_scale(1, 1:(numel(idxU_obs)+numel(idxV_obs)+numel(idxP_obs))) =...
        [Vel_state, Pres_state];
    
    obs = atan(state_scale);
    obs = obs+ randn(size(state_EnSF(i+1, :))) * obs_sigma;
    for l =1:5
        Vel_EnSF = x0_EnSF(:, 1:(numel(idxU_obs)+numel(idxV_obs)));
        Pres_EnSF = ...
            x0_EnSF(:, numel(idxU_obs)+numel(idxV_obs)+(1:numel(idxP_obs)));
    
        mask_pres1_EnSF = ( (-1e-1 <= Pres_EnSF) & (Pres_EnSF < -1e-2) ) |...
            ( (1e-2 <= Pres_EnSF) & (Pres_EnSF < 1e-1) );
    
        mask_pres2_EnSF = ( (-1e-2 <= Pres_EnSF) & (Pres_EnSF < -1e-3) ) |...
            ( (1e-3 <= Pres_EnSF) & (Pres_EnSF < 1e-2) );
    
        mask_pres3_EnSF = ( (-1e-3 <= Pres_EnSF) & (Pres_EnSF < -1e-4) ) |...
            ( (1e-4 <= Pres_EnSF) & (Pres_EnSF < 1e-3) );
    
        mask_pres4_EnSF = ( (-1e-4 <= Pres_EnSF) & (Pres_EnSF < -1e-5) ) |...
            ( (1e-5 <= Pres_EnSF) & (Pres_EnSF < 1e-4) );
    
        mask_pres5_EnSF = ( (-1e-5 <= Pres_EnSF) & (Pres_EnSF < -1e-6) ) |...
            ( (1e-6 <= Pres_EnSF) & (Pres_EnSF < 1e-5) );
    
        mask_pres6_EnSF = ( (-1e-6 <= Pres_EnSF) & (Pres_EnSF < -1e-7) ) |...
            ( (1e-7 <= Pres_EnSF) & (Pres_EnSF < 1e-6) );
    
        mask_pres7_EnSF = ( (-1e-7 <= Pres_EnSF) & (Pres_EnSF < -1e-8) ) |...
            ( (1e-8 <= Pres_EnSF) & (Pres_EnSF < 1e-7) );
        mask_pres8_EnSF = ( (-1e-8 <= Pres_EnSF) & (Pres_EnSF < -1e-9) ) |...
            ( (1e-9 <= Pres_EnSF) & (Pres_EnSF < 1e-8) );
    
        mask_pres9_EnSF = ( (-1e-9 <= Pres_EnSF) & (Pres_EnSF < -1e-10) ) |...
            ( (1e-10 <= Pres_EnSF) & (Pres_EnSF < 1e-9) );
        mask_pres10_EnSF = ( (-1e-10 <= Pres_EnSF) & (Pres_EnSF < -1e-11) ) |...
            ( (1e-11 <= Pres_EnSF) & (Pres_EnSF < 1e-10) );
        mask_pres11_EnSF = ( (-1e-11 <= Pres_EnSF) & (Pres_EnSF < -1e-12) ) |...
                ( (1e-12 <= Pres_EnSF) & (Pres_EnSF < 1e-11) );
        mask_pres12_EnSF = ( (-1e-12 <= Pres_EnSF) & (Pres_EnSF < -1e-13) ) |...
                ( (1e-13 <= Pres_EnSF) & (Pres_EnSF < 1e-12) );
        mask_pres13_EnSF = ( (-1e-13 <= Pres_EnSF) & (Pres_EnSF < -1e-14) ) |...
                ( (1e-14 <= Pres_EnSF) & (Pres_EnSF < 1e-13) );
        mask_pres14_EnSF = ( (-1e-14 <= Pres_EnSF) & (Pres_EnSF < -1e-15) ) |...
                ( (1e-15 <= Pres_EnSF) & (Pres_EnSF < 1e-14) );
        mask_pres15_EnSF = ( (-1e-15 <= Pres_EnSF) & (Pres_EnSF < -1e-16) ) |...
                ( (1e-16 <= Pres_EnSF) & (Pres_EnSF < 1e-15) );
        mask_pres16_EnSF = ( (-1e-16 <= Pres_EnSF) & (Pres_EnSF < -1e-17) ) |...
                ( (1e-17 <= Pres_EnSF) & (Pres_EnSF < 1e-16) );
        mask_pres17_EnSF = ( (-1e-17 <= Pres_EnSF) & (Pres_EnSF < 0) ) |...
                ( (0 <= Pres_EnSF) & (Pres_EnSF < 1e-17) );
    
        [row_pres1, col_pres1] = find(mask_pres1_EnSF);
        [row_pres2, col_pres2] = find(mask_pres2_EnSF);
        [row_pres3, col_pres3] = find(mask_pres3_EnSF);
        [row_pres4, col_pres4] = find(mask_pres4_EnSF);
        [row_pres5, col_pres5] = find(mask_pres5_EnSF);
        [row_pres6, col_pres6] = find(mask_pres6_EnSF);
        [row_pres7, col_pres7] = find(mask_pres7_EnSF);
        [row_pres8, col_pres8] = find(mask_pres8_EnSF);
        [row_pres9, col_pres9] = find(mask_pres9_EnSF);
        [row_pres10, col_pres10] = find(mask_pres10_EnSF);
        [row_pres11, col_pres11] = find(mask_pres11_EnSF);
        [row_pres12, col_pres12] = find(mask_pres12_EnSF);
        [row_pres13, col_pres13] = find(mask_pres13_EnSF);
        [row_pres14, col_pres14] = find(mask_pres14_EnSF);
        [row_pres15, col_pres15] = find(mask_pres15_EnSF);
        [row_pres16, col_pres16] = find(mask_pres16_EnSF);
        [row_pres17, col_pres17] = find(mask_pres17_EnSF);
    
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres1, col_pres1)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres1, col_pres1))*1e1;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres2, col_pres2)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres2, col_pres2))*1e2;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres3, col_pres3)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres3, col_pres3))*1e3;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres4, col_pres4)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres4, col_pres4))*1e4;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres5, col_pres5)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres5, col_pres5))*1e5;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres6, col_pres6)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres6, col_pres6))*1e6;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres7, col_pres7)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres7, col_pres7))*1e7;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres8, col_pres8)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres8, col_pres8))*1e8;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres9, col_pres9)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres9, col_pres9))*1e9;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres10, col_pres10)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres10, col_pres10))*1e10;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres11, col_pres11)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres11, col_pres11))*1e11;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres12, col_pres12)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres12, col_pres12))*1e12;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres13, col_pres13)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres13, col_pres13))*1e13;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres14, col_pres14)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres14, col_pres14))*1e14;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres15, col_pres15)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres15, col_pres15))*1e15;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres16, col_pres16)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres16, col_pres16))*1e16;
        Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres17, col_pres17)) ...
            = Pres_EnSF(sub2ind(size(Pres_EnSF), row_pres17, col_pres17))*1e17;
    
        x0_EnSF(:, 1:(numel(idxU_obs)+numel(idxV_obs)+numel(idxP_obs))) = ...
            [Vel_EnSF, Pres_EnSF];
        
        sln_bar = reverse_SDE(x0_EnSF, obs, euler_steps, 1, 1,...
            false, eps_alpha, obs_sigma);
        
        Vel_EnSF = sln_bar(:, 1:(numel(idxU_obs)+numel(idxV_obs)));
        Pres_EnSF = ...
            sln_bar(:, (numel(idxU_obs)+numel(idxV_obs))+(1:numel(idxP_obs)));
    
        Pres_EnSF(:, indob_presscale1) = Pres_EnSF(:, indob_presscale1)/1e1;
        Pres_EnSF(:, indob_presscale2) = Pres_EnSF(:, indob_presscale2)/1e2;
        Pres_EnSF(:, indob_presscale3) = Pres_EnSF(:, indob_presscale3)/1e3;
        Pres_EnSF(:, indob_presscale4) = Pres_EnSF(:, indob_presscale4)/1e4;
        Pres_EnSF(:, indob_presscale5) = Pres_EnSF(:, indob_presscale5)/1e5;
        Pres_EnSF(:, indob_presscale6) = Pres_EnSF(:, indob_presscale6)/1e6;
        Pres_EnSF(:, indob_presscale7) = Pres_EnSF(:, indob_presscale7)/1e7;
        Pres_EnSF(:, indob_presscale8) = Pres_EnSF(:, indob_presscale8)/1e8;
        Pres_EnSF(:, indob_presscale9) = Pres_EnSF(:, indob_presscale9)/1e9;
        Pres_EnSF(:, indob_presscale10) = ...
            Pres_EnSF(:, indob_presscale10)/1e10;
        Pres_EnSF(:, indob_presscale11) = ...
            Pres_EnSF(:, indob_presscale11)/1e11;
        Pres_EnSF(:, indob_presscale12) = ...
            Pres_EnSF(:, indob_presscale12)/1e12;
        Pres_EnSF(:, indob_presscale13) = ...
            Pres_EnSF(:, indob_presscale13)/1e13;
        Pres_EnSF(:, indob_presscale14) = ...
            Pres_EnSF(:, indob_presscale14)/1e14;
        Pres_EnSF(:, indob_presscale15) = ...
            Pres_EnSF(:, indob_presscale15)/1e15;
        Pres_EnSF(:, indob_presscale16) = ...
            Pres_EnSF(:, indob_presscale16)/1e16;
        Pres_EnSF(:, indob_presscale17) = ...
            Pres_EnSF(:, indob_presscale17)/1e17;
    
        x0_EnSF(:, 1:(numel(idxU_obs)+numel(idxV_obs)+numel(idxP_obs))) = ...
            [Vel_EnSF, Pres_EnSF];
    end
    x_state(:, idx_obs) = x0_EnSF;
    % cols = (Size_U + Size_V) + (1:Size_P); 
    % if i < 400
    %     x_state(:, 1:Size_U) = min( max( x_state(:, 1:Size_U), -0.25), 0.25);
    %     x_state(:, Size_U+(1:Size_V)) =...
    %         min( max( x_state(:, Size_U+(1:Size_V)), -0.5), 0.5);
    % else
    %     x_state(:, 1:Size_U) = min( max( x_state(:, 1:Size_U), -0.35), 0.35);
    %     x_state(:, Size_U+(1:Size_V)) =...
    %         min( max( x_state(:, Size_U+(1:Size_V)), -0.33), 0.33);
    % end
    
    Unew = x_state(:, 1:Size_U);
    Vnew = x_state(:, Size_U+(1:Size_V));
    
    Unew = reshape(Unew', [mU0, nU0, ensemble_size]);
    Vnew = reshape(Vnew', [mV0, nV0, ensemble_size]);
    
    Uold = reshape(U0_state', [mU0, nU0, ensemble_size]);
    Vold = reshape(V0_state', [mV0, nV0, ensemble_size]);
    
    q_update = Update_q_EnSF(q0_batch, Unew, Vnew, Uold, Vold,...
            hx, hy, dtEnSF, theta);
    
    U00_state = U0_state;
    V00_state = V0_state;
    
    U0_state = x_state(:, 1:Size_U);
    V0_state = x_state(:, Size_U+(1:Size_V));

    est_save(2, :) = est_save(2, :) + mean(x_state, 1);
    
    q00_batch = q0_batch;
    q0_batch = q_update;
end
