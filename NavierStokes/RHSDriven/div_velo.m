function [Udx,Vdy] = div_velo(U,V,hx,hy)
    % U and V satisfy: [U;V].n = 0
    U_ext = zeros(size(U)+[2 0]); 
    U_ext(2:end-1,:) = U;
    Udx = diff(U_ext)/hx;
    V_ext = zeros(size(V)+[0 2]); 
    V_ext(:,2:end-1) = V;
    Vdy = diff(V_ext')'/hy;
end