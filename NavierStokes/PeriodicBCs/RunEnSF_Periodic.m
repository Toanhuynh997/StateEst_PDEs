%% Run this file to compare the results with the reference solution
clear
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
load cmap.mat cmap
load('TestRefSol_BDF2_Periodic.mat', 'U_Py', 'V_Py', 'P_Py')

%% EnSF
% load('ResultEnSF_Periodic_T100_100Obs_noise0001_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'P_EnSF', 'q_EnSF')
% load('ResultEnSF_Periodic_T100_100Obs_noise01_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'P_EnSF', 'q_EnSF')

% load('ResultEnSF_Periodic_T100_70Obs_noise0001_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'P_EnSF', 'q_EnSF')
% load('ResultEnSF_Periodic_T100_70Obs_noise01_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'P_EnSF', 'q_EnSF')

% load('ResultEnSF_Periodic_T100_7Obs_noise0001_BiH_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'P_EnSF', 'q_EnSF')
% load('ResultEnSF_Periodic_T100_7Obs_noise01_BiH_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'P_EnSF', 'q_EnSF')


%% LETKF
% load('ResultLETKF_Periodic_T100_7Obs_noise0001_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'P_EnSF')

% load('ResultLETKF_Periodic_T100_7Obs_noise01_v1.mat', 'U_EnSF',...
%     'V_EnSF', 'P_EnSF')

U_EnSF = U_EnSF';
V_EnSF = V_EnSF';
P_EnSF = P_EnSF';
egy_EnSF = zeros(NtEnSF+1, 1);

U = U_Py';
V = V_Py';
P = P_Py';
egy_Ref = zeros(NtEnSF+1, 1);

URef = reshape(U(:, end), Nx, Ny);
VRef = reshape(V(:, end), Nx, Ny);
PRef = reshape(P(:, end), Nx, Ny);

for ll = 1:(NtEnSF+1)
    U = reshape(U_EnSF0001(:, ll), Nx, Ny);
    V = reshape(V_EnSF0001(:, ll), Nx, Ny);
    egy_EnSF(ll) = 0.5*hx*hy*(sum(sum(U.*U))+sum(sum(V.*V)));
end


U_EnSF = reshape(U_EnSF(:, end), Nx, Ny);
V_EnSF = reshape(V_EnSF(:, end), Nx, Ny);
P_EnSF = reshape(P_EnSF(:, end), Nx, Ny);
% [U,V,P,q, qq, egy, egy_theta, perS, Xp, Yp] = ...
%     NS_bEuler_per(xa,xb,ya,yb,T,Nx,Ny,Nt,mu,theta,opt);

[~, ~, ~, ~, qq, egy, perS, Xp, Yp] =...
    NS_BDF_per(xa,xb,ya,yb,T,Nx,Ny,NtRef,mu,theta,opt);

figure
plot(egy)
hold on
plot(egy_EnSF)
legend('Reference', 'Estimate')

URef_aug = [URef; zeros(1, Ny)];
VRef_aug = [VRef zeros(Nx, 1)];

URef_average = (URef_aug(1:end-1, :)+URef_aug(2:end, :))/2;
VRef_average = (VRef_aug(:, 1:end-1)+VRef_aug(:, 2:end))/2;

UEnSF_aug = [U_EnSF; zeros(1, Ny)];
VEnSF_aug = [V_EnSF zeros(Nx, 1)];

UEnSF_average = (UEnSF_aug(1:end-1, :)+UEnSF_aug(2:end, :))/2;
VEnSF_average = (VEnSF_aug(:, 1:end-1)+VEnSF_aug(:, 2:end))/2;

velo = sqrt(URef_average.^2+VRef_average.^2);

veloEnSF = sqrt(UEnSF_average.^2+VEnSF_average.^2);
% 
levels = 0:0.1:1;
figure
contourf(xmid,ymid,velo,levels,'LineStyle','none');
hold on;
[C,h] = contour(xmid,ymid,velo,levels,'LineColor','k','linewidth',2);
clabel(C,h,'FontWeight','bold');
hold off;
axis square; 
colormap(cmap);
colorbar; 
xlabel('x'); ylabel('y');
% title('The velocity field');
set(gca,'FontWeight','bold','LineWidth',2,'FontSize',20)
% 
levels = 0:0.1:1;
figure
contourf(xmid,ymid,veloEnSF,levels,'LineStyle','none');
hold on;
[CEnSF,h] = contour(xmid,ymid,veloEnSF,levels,'LineColor','k','linewidth',2);
clabel(CEnSF,h,'FontWeight','bold');
hold off;
axis square; 
colormap(cmap);
colorbar; 
xlabel('x'); ylabel('y');
% title('The velocity field');
set(gca,'FontWeight','bold','LineWidth',2,'FontSize',20)

figure(6)
quiver(Xp, Yp, URef_average, VRef_average);
% 
figure(7)
quiver(Xp, Yp, UEnSF_average, VEnSF_average);

figure; 
contourf(Xp,Yp,PRef);
colormap(jet);
shading interp
% 
figure; 
contourf(Xp,Yp,P_EnSF);
colormap(jet);
shading interp

figure
surf(P_EnSF)