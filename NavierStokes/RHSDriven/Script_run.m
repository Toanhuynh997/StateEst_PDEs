%% Run this file for the reference solution
clear
xa = 0; xb = 1; 
ya = 0; yb = 1; 

T = 15; 

mu_small = 1/1000; theta = 100; 

Nx = 80; 

Ny = Nx; 
Nt_coarse = 1000;

U0 = zeros(Nx, Ny);
V0 = zeros(Nx, Ny-1);

mu_big = 1/2000; 
Nt_fine = 8000;

hx = (xb-xa)/Nx; 
hy = (yb-ya)/Ny;
x = xa:hx:xb; 
y = ya:hy:yb;
xmid = avg(x);
ymid = avg(y);
dt_fine = T/Nt_fine;    
TTEnSF = 0:dt_fine:T;

alpha = 1/dt_fine;

sU = Nx*Ny; 
sV = Nx*(Ny-1); 
sP = Nx*Ny; 

% [Yu,Xu] = meshgrid(ymid,x(1:end-1));
% [Yv,Xv] = meshgrid(y(2:end-1),xmid);
% [Yp,Xp] = meshgrid(ymid,xmid);

% [U,V,P,q, Xp, Yp, Xu, Yu, Xv, Yv, perS_BE, qq, egy] = ....
%     NS_bEuler_ChannelFlow_NoCyl(U0, V0, xa,xb,ya,yb,T,Nx,Ny,Nt_fine,...
%     mu_big,theta, 1);

[U,V,P,q, Xp, Yp, Xu, Yu, Xv, Yv, ~, qq, egy] = ....
    NS_BDF2_ChannelFlow_NoCyl_v1(U0, V0, xa,xb,ya,yb, dt_fine, T,Nx,Ny,Nt_fine,...
    mu_big,theta, 1);

U_Ref = U;
V_Ref = V; 
P_Ref = P;

% % 

save('RefSol_BDF2_Mixed_PeriDiri_80_8000_opt1_Re2000v1.mat', 'U_Ref', ...
    'V_Ref', 'P_Ref', 'q_Ref')

Uplot = reshape(U_Ref(end, :), Nx, Ny);
Vplot = reshape(V_Ref(end, :), Nx, Ny-1);
Pplot = reshape(P_Ref(end, :), Nx, Ny);

U_aug = [Uplot; Uplot(1, :)];
V_aug = [zeros(Nx, 1) Vplot zeros(Nx, 1)];

U_average = (U_aug(1:end-1, :)+U_aug(2:end, :))/2;
V_average = (V_aug(:, 1:end-1)+V_aug(:, 2:end))/2;
% 
figure
quiver(Xp, Yp, U_average, V_average);

figure
contourf(Xp,Yp,Pplot);
colormap(jet);
shading interp

velo = sqrt(U_average.^2+V_average.^2);

levels = 0:0.1:1;
figure
contourf(Xp,Yp,velo,levels,'LineStyle','none');
hold on;
[C,h] = contour(Xp,Yp,velo,levels,'LineColor','k','linewidth',2);
clabel(C,h,'FontWeight','bold');
hold off;
axis square; 
% colormap(cmap);
colorbar; 
xlabel('x'); ylabel('y');
% title('The velocity field');
set(gca,'FontWeight','bold','LineWidth',2,'FontSize',20)

figure
plot(q_Ref)