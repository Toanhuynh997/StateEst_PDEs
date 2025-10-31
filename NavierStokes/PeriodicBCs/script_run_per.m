%% NSE with periodic BCs
clear
% Domain (0,1)x(0,1)
xa = 0; xb = 1; ya = 0; yb = 1; T = 1;
mu = 0.001; theta = 5; opt = 2;
Nx = 40; Ny = Nx; 

% Nt = T*100;

Nt = T*600;

%% Run this one to get the perS for the solver with BE
% [U,V,P,q, qq, egy, egy_theta, perS, Xp, Yp] = ...
%     NS_bEuler_per(xa,xb,ya,yb,T,Nx,Ny,Nt,mu,theta,opt);

% save('Permutation_Indices_RefSol_Per40.mat', 'perS') 

%% with Nt = 100 for the EnSF algorithm
% save('Permutation_Indices_EnSF_Per40_T100.mat', 'perS')

%% Run this one to get the perS for the solver with BDF2
[U, V, P,~, qq, egy, perS, Xp, Yp] =...
    NS_BDF_per(xa,xb,ya,yb,T,Nx,Ny,Nt,mu,theta,opt);

save('Permutation_Indices_RefSol_BDF2Per40.mat', 'perS')

% save('Permutation_Indices_EnSF_BDF2Per40.mat', 'perS')

U_aug = [U; zeros(1, Ny)];
V_aug = [V zeros(Nx, 1)];

U_average = (U_aug(1:end-1, :)+U_aug(2:end, :))/2;
V_average = (V_aug(:, 1:end-1)+V_aug(:, 2:end))/2;

figure
quiver(Xp, Yp, U_average, V_average);

figure; 
contourf(Xp,Yp,P);
colormap(jet);
shading interp

figure
surf(Xp, Yp, P)



























%% Backward Euler
% Plot L2 errors for # theta (opt = 1, mu = 0.1, Nx = Ny = 400)
% dt = log10([1/5 1/10 1/20 1/40]);
% 
% %================================================================== 
% % mu = 0.1
% % L2err_u_theta100inv = log10([1.33e-02 7.14e-03 4.41e-03 3.50e-03]);
% % L2err_p_theta100inv = log10([1.22e-01 1.28e-01 1.39e-01 1.48e-01]);
% % 
% % L2err_u_theta10inv = log10([1.32e-02 6.85e-03 3.67e-03 2.20e-03]);
% % L2err_p_theta10inv = log10([1.04e-01 8.45e-02 7.63e-02 7.31e-02]);
% % 
% % L2err_u_theta1 = log10([1.32e-02 6.73e-03 3.41e-03 1.74e-03]);
% % L2err_p_theta1 = log10([9.57e-02 5.84e-02 3.53e-02 2.25e-02]);
% % 
% % L2err_u_theta10 = log10([1.32e-02 6.72e-03 3.39e-03 1.71e-03]);
% % L2err_p_theta10 = log10([9.45e-02 5.48e-02 2.99e-02 1.59e-02]);
% % 
% % L2err_u_theta100 = log10([1.32e-02 6.72e-03 3.39e-03 1.71e-03]);
% % L2err_p_theta100 = log10([9.44e-02 5.45e-02 2.93e-02 1.52e-02]);
% % 
% % % L2err_u_theta1000 = log10([1.32e-02 6.72e-03 3.39e-03 1.70e-03]);
% % % L2err_p_theta1000 = log10([9.44e-02 5.44e-02 2.92e-02 1.52e-02]);
% 
% %================================================================== 
% % mu = 0.001
% L2err_u_theta100inv = log10([6.93e-02 3.46e-02 1.73e-02 8.61e-03]);
% L2err_p_theta100inv = log10([6.72e-02 5.32e-02 3.10e-02 1.40e-02]);
% 
% L2err_u_theta10inv = log10([6.96e-02 3.48e-02 1.74e-02 8.71e-03]);
% L2err_p_theta10inv = log10([4.17e-02 2.13e-02 1.11e-02 6.03e-03]);
% 
% L2err_u_theta1 = log10([7.01e-02 3.50e-02 1.75e-02 8.75e-03]);
% L2err_p_theta1 = log10([6.02e-02 3.18e-02 1.64e-02 8.33e-03]);
% 
% L2err_u_theta10 = log10([7.02e-02 3.51e-02 1.75e-02 8.75e-03]);
% L2err_p_theta10 = log10([6.22e-02 3.29e-02 1.69e-02 8.57e-03]);
% 
% L2err_u_theta100 = log10([7.02e-02 3.51e-02 1.75e-02 8.75e-03]);
% L2err_p_theta100 = log10([6.24e-02 3.30e-02 1.70e-02 8.59e-03]);
% %================================================================== 
% 
% s = 2; MKsize = 20;
% u_start = min([L2err_u_theta100(1) L2err_u_theta10(1) L2err_u_theta1(1) ...
%                L2err_u_theta10inv(1) L2err_u_theta100inv(1)]);
% p_start = min([L2err_p_theta100(1) L2err_p_theta10(1) L2err_p_theta1(1) ...
%                L2err_p_theta10inv(1) L2err_p_theta100inv(1)]);
% figure
% plot(dt,u_start-0.3-log10(2)*[0 1 2 3],'--k','linewidth',s)
% hold on
% plot(dt,L2err_u_theta100inv,'-r','linewidth',s,'Marker','+','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_u_theta10inv,'-b','linewidth',s,'Marker','*','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_u_theta1,'-g','linewidth',s,'Marker','x','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_u_theta10,'-c','linewidth',s,'Marker','s','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_u_theta100,'-m','linewidth',s,'Marker','d','MarkerSize',MKsize)
% % hold on
% % plot(dt,L2err_u_theta1000,'-y','linewidth',s,'Marker','o','MarkerSize',MKsize)
% xlim([-1.7 -0.6])
% % ylim([-3 -1.7])
% ylim([-5.2 -0.5])
% xlabel('log_{10}(\tau)')
% ylabel('log_{10}(Error)')
% title('L^2 errors of velocity')
% legend('Slope = 1','\theta = 0.01','\theta = 0.1','\theta = 1','\theta = 10','\theta = 100','Location','southeast')
% set(gca,'FontWeight','bold','LineWidth',s,'FontSize',18)
% 
% figure
% plot(dt,p_start-0.3-log10(2)*[0 1 2 3],'--k','linewidth',s)
% hold on
% plot(dt,L2err_p_theta100inv,'-r','linewidth',s,'Marker','+','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_p_theta10inv,'-b','linewidth',s,'Marker','*','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_p_theta1,'-g','linewidth',s,'Marker','x','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_p_theta10,'-c','linewidth',s,'Marker','s','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_p_theta100,'-m','linewidth',s,'Marker','d','MarkerSize',MKsize)
% % hold on
% % plot(dt,L2err_p_theta1000,'-y','linewidth',s,'Marker','o','MarkerSize',MKsize)
% xlim([-1.7 -0.6])
% % ylim([-2.2 -0.6])
% ylim([-5.2 -0.5])
% set(gca,'FontWeight','bold','LineWidth',s,'FontSize',18)
% xlabel('log_{10}(\tau)')
% ylabel('log_{10}(Error)')
% title('L^2 errors of pressure')
% legend('Slope = 1','\theta = 0.01','\theta = 0.1','\theta = 1','\theta = 10','\theta = 100','Location','southeast')
% 
% %% Crank-Nicolson
% % Plot L2 errors for # theta (opt = 1, mu = 0.1, Nx = Ny = 400)
% dt = log10([1/5 1/10 1/20 1/40]);
% %================================================================== 
% % mu = 0.1
% % L2err_u_theta100inv = log10([4.07e-03 2.52e-03 2.44e-03 2.44e-03]);
% % L2err_p_theta100inv = log10([1.08e-01 1.23e-01 1.26e-01 1.27e-01]);
% % 
% % L2err_u_theta10inv = log10([3.54e-03 9.60e-04 4.46e-04 3.92e-04]);
% % L2err_p_theta10inv = log10([2.69e-02 2.24e-02 2.16e-02 2.15e-02]);
% % 
% % L2err_u_theta1 = log10([3.51e-03 8.72e-04 2.18e-04 6.46e-05]);
% % L2err_p_theta1 = log10([1.58e-02 5.47e-03 3.07e-03 2.47e-03]);
% % 
% % L2err_u_theta10 = log10([3.51e-03 8.69e-04 2.11e-04 4.76e-05]);
% % L2err_p_theta10 = log10([1.47e-02 3.68e-03 1.10e-03 4.48e-04]);
% % 
% % L2err_u_theta100 = log10([3.51e-03 8.69e-04 2.11e-04 4.71e-05]);
% % L2err_p_theta100 = log10([1.46e-02 3.50e-03 8.97e-04 2.45e-04]);
% % 
% % % L2err_u_theta1000 = log10([3.51e-03 8.69e-04 2.11e-04 4.71e-05]);
% % % L2err_p_theta1000 = log10([1.45e-02 3.48e-03 8.77e-04 2.25e-04]);
% 
% %================================================================== 
% % mu = 0.001
% L2err_u_theta100inv = log10([2.40e-03 6.86e-04 2.86e-04 1.90e-04]);
% L2err_p_theta100inv = log10([1.26e-02 4.54e-03 2.80e-03 2.37e-03]);
% 
% L2err_u_theta10inv  = log10([2.31e-03 5.61e-04 1.48e-04 4.81e-05]);
% L2err_p_theta10inv  = log10([1.16e-02 2.79e-03 8.73e-04 3.94e-04]);
% 
% L2err_u_theta1      = log10([2.30e-03 5.48e-04 1.35e-04 3.44e-05]);
% L2err_p_theta1      = log10([1.15e-02 2.62e-03 6.79e-04 1.94e-04]);
% 
% L2err_u_theta10     = log10([2.30e-03 5.47e-04 1.34e-04 3.30e-05]);
% L2err_p_theta10     = log10([1.14e-02 2.60e-03 6.59e-04 1.74e-04]);
% 
% L2err_u_theta100    = log10([2.30e-03 5.47e-04 1.33e-04 3.29e-05]);
% L2err_p_theta100    = log10([1.14e-02 2.60e-03 6.57e-04 1.72e-04]);
% %================================================================== 
% 
% s = 2; MKsize = 20;
% figure
% plot(dt,L2err_u_theta100(1)-0.3-2*log10(2)*[0 1 2 3],'--k','linewidth',s)
% hold on
% plot(dt,L2err_u_theta100inv,'-r','linewidth',s,'Marker','+','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_u_theta10inv,'-b','linewidth',s,'Marker','*','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_u_theta1,'-g','linewidth',s,'Marker','x','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_u_theta10,'-c','linewidth',s,'Marker','s','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_u_theta100,'-m','linewidth',s,'Marker','d','MarkerSize',MKsize)
% % hold on
% % plot(dt,L2err_u_theta1000,'-y','linewidth',s,'Marker','o','MarkerSize',MKsize)
% xlim([-1.7 -0.6])
% % ylim([-4.8 -2.2])
% ylim([-5.2 -0.5])
% xlabel('log_{10}(\tau)')
% ylabel('log_{10}(Error)')
% title('L^2 errors of velocity')
% legend('Slope = 2','\theta = 0.01','\theta = 0.1','\theta = 1','\theta = 10','\theta = 100','Location','southeast')
% set(gca,'FontWeight','bold','LineWidth',s,'FontSize',18)
% 
% figure
% plot(dt,L2err_p_theta100(1)-0.3-2*log10(2)*[0 1 2 3],'--k','linewidth',s)
% hold on
% plot(dt,L2err_p_theta100inv,'-r','linewidth',s,'Marker','+','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_p_theta10inv,'-b','linewidth',s,'Marker','*','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_p_theta1,'-g','linewidth',s,'Marker','x','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_p_theta10,'-c','linewidth',s,'Marker','s','MarkerSize',MKsize)
% hold on
% plot(dt,L2err_p_theta100,'-m','linewidth',s,'Marker','d','MarkerSize',MKsize)
% % hold on
% % plot(dt,L2err_p_theta1000,'-y','linewidth',s,'Marker','o','MarkerSize',MKsize)
% xlim([-1.7 -0.6])
% % ylim([-4.4 -0.4])
% ylim([-5.2 -0.5])
% set(gca,'FontWeight','bold','LineWidth',s,'FontSize',18)
% xlabel('log_{10}(\tau)')
% ylabel('log_{10}(Error)')
% title('L^2 errors of pressure')
% legend('Slope = 2','\theta = 0.01','\theta = 0.1','\theta = 1','\theta = 10','\theta = 100','Location','southeast')
% 
% 
% %% Plot q
% load CN_Taylor_Green_T5_dt0025_NxNy200_mu001_theta100.mat egy egy_theta qq ...
%                          xa xb ya yb T mu theta opt Nx Ny Nt
% qqq = qq-1;
% TT = 0:0.025:5;
% figure
% plot(TT,qqq,LineWidth=2)
% ylim(1e-13*[-3 3])
% % legend('\boldmath$\mathcal{K}$','\boldmath$\mathcal{K}_{\theta}$','\boldmath$\mathcal{K}_{exact}$','Interpreter','latex')
% xlabel('\bf{Time}')
% ylabel('\bf{q}')
% set(gca, 'FontWeight', 'bold', 'linewidth',1.5,'fontsize',20)