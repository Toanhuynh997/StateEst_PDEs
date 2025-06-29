function [UgradU1,UgradU2] = compute_UgradU(U,V,hx,hy)
    %% uu_x+vu_y: u_x-periodic; u_y-Dirichlet
    
    %%%% Compute V on the U_grid: take the average of four adjacent dofs
    Udx = [U(end,:);U;U(1,:)];
    Udx = (Udx(3:end,:)-Udx(1:end-2,:))/(2*hx);

    Udy = zeros(size(U));
    Udy(:,2:end-1) = (U(:,3:end)-U(:,1:end-2))/(2*hy);
    Udy(:,1) = (U(:,2)+U(:,1))/(2*hy);
    Udy(:,end) = -(U(:,end)+U(:,end-1))/(2*hy);
    
    %%%%% Augment V
    Vmod = zeros(size(V,1),size(V,2)+2); 
    Vmod(:,2:end-1) = V;
    Vmod = [Vmod(end, :); Vmod];
    Vmod = avg(Vmod); Vmod = avg(Vmod')';
    UgradU1 = U.*Udx + Vmod.*Udy;
    
    %% uv_x+vv_y: v_x-periodic; v_y-Dirichlet
    
    Vdx = [V(end,:);V;V(1,:)];
    Vdx = (Vdx(3:end,:)-Vdx(1:end-2,:))/(2*hx); % Udx~U
    
    col0 = zeros(size(V,1),1);
    Vdy = ([V(:,2:end), col0]-[col0, V(:,1:end-1)])/(2*hy);
    
    %%%%% Augment U
    Umod = [U;U(1,:)];
    Umod = avg(Umod); 
    Umod = avg(Umod')';
    UgradU2 = Umod.*Vdx + V.*Vdy;
end