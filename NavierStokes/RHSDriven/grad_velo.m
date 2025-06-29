function [Udx,Udy] = grad_velo(U,hx,hy)
    Udx = [U(end,:);U;U(1,:)];
    Udx = (Udx(3:end,:)-Udx(1:end-2,:))/(2*hx); % Udx~U
    Udy = [U(:,end),U,U(:,1)];
    Udy = (Udy(:,3:end)-Udy(:,1:end-2))/(2*hy); % Udy~U
end