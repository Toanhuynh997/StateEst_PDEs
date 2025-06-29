clear
%% Run this file to compare the results with different amount of observations
xa = 0; xb = 1; ya = 0; yb = 1; T = 1;
mu = 0.001; theta = 5; opt = 2;
Nx = 40; Ny = Nx; 
hx = (xb-xa)/Nx;
hy = (yb-ya)/Ny;
x = xa:hx:xb;
y = ya:hy:yb;

xmid = (x(2:end)+x(1:end-1))/2;
ymid = (y(2:end)+y(1:end-1))/2;

[YY, XX] = meshgrid(y, x);

NtRef = T*100;
NtEnSF = T*100;
[~, ~, ~, ~, qq, egy, perS, Xp, Yp] =...
    NS_BDF_per(xa,xb,ya,yb,T,Nx,Ny,NtRef,mu,theta,opt);
% 
% load('ResultEnSF_Periodic_T100_7Obs_noise0001_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'rmse')
load('ResultEnSF_Periodic_T100_7Obs_noise01_v1.mat', 'U_EnSF',...
    'V_EnSF', 'rmse')
rmse_NoIP = rmse;
U_EnSF_NoIP = U_EnSF';
V_EnS_NoIP = V_EnSF';

egy_EnSF_NoIP = zeros(NtEnSF+1, 1);
for ll = 1:(NtEnSF+1)
    U = reshape(U_EnSF_NoIP(:, ll), Nx, Ny);
    V = reshape(V_EnS_NoIP(:, ll), Nx, Ny);
    egy_EnSF_NoIP(ll) = 0.5*hx*hy*(sum(sum(U.*U))+sum(sum(V.*V)));
end

% load('ResultEnSF_Periodic_T100_7Obs_noise0001_BiH_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'rmse')
load('ResultEnSF_Periodic_T100_7Obs_noise01_BiH_v1.mat', 'U_EnSF',...
    'V_EnSF', 'rmse')

rmse_BiH = rmse;

U_EnSF_BiH = U_EnSF';
V_EnS_BiH = V_EnSF';
egy_EnSF_BiH = zeros(NtEnSF+1, 1);
for ll = 1:(NtEnSF+1)
    U = reshape(U_EnSF_BiH(:, ll), Nx, Ny);
    V = reshape(V_EnS_BiH(:, ll), Nx, Ny);
    egy_EnSF_BiH(ll) = 0.5*hx*hy*(sum(sum(U.*U))+sum(sum(V.*V)));
end

% load('ResultLETKF_Periodic_T100_7Obs_noise0001_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'rmse')
load('ResultLETKF_Periodic_T100_7Obs_noise01_v1.mat', 'U_EnSF',...
    'V_EnSF', 'rmse')

rmse_LETKF = rmse;
U_EnSF_LETKF = U_EnSF';
V_EnS_LETKF = V_EnSF';

egy_EnSF_LETKF = zeros(NtEnSF+1, 1);
for ll = 1:(NtEnSF+1)
    U = reshape(U_EnSF_LETKF(:, ll), Nx, Ny);
    V = reshape(V_EnS_LETKF(:, ll), Nx, Ny);
    egy_EnSF_LETKF(ll) = 0.5*hx*hy*(sum(sum(U.*U))+sum(sum(V.*V)));
end


figure
plot(rmse_BiH)
hold on
plot(rmse_NoIP)
% legend('With Inpainting', 'No Inpainting')
hold on
plot(rmse_LETKF)
legend('With Inpainting', 'No Inpainting','LETKF')

figure
plot(egy_EnSF_BiH)
hold on
plot(egy_EnSF_NoIP)
% hold on
% plot(egy)
% legend('With Inpainting', 'No Inpainting', 'Reference')
hold on
plot(egy_EnSF_LETKF)
hold on
plot(egy)
legend('With Inpainting', 'No Inpainting', 'LETKF', 'Reference')