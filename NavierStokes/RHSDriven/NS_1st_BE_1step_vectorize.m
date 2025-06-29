function [U_new, V_new, P_new, q_new] = NS_1st_BE_1step_vectorize(...
    hx, hy, dt, U, V, q, mu, theta, DiE, BDiE, DiBt, Di, B, Bt,...
    perS, SS, SSt, Nx, Ny, sU,  alpha, A1, ff1, ff2)


    rhs1 = alpha.* U + ff1;      % shape [Nx-1, Ny, L] (assuming U has that shape)
    rhs2 = alpha.* V + ff2; 


    rhs1_flat = reshape(rhs1, [(size(rhs1,1)*size(rhs1,2)), ...
        size(rhs1,3)]);  % MATLAB is col-major
    rhs2_flat = reshape(rhs2, [(size(rhs2,1)*size(rhs2,2)),...
        size(rhs2,3)]);
    
    rhs_full = [rhs1_flat; rhs2_flat];  % shape => ( (Nx-1)*Ny + Nx*(Ny-1), L )
    
    % Solve stokes system (batch)
    [U1, V1, P1] = ...
        stokes_solver_vectorized(rhs_full, DiE, BDiE, DiBt, Di, B, Bt, ...
                                 perS, SS, SSt, Nx, Ny, sU);


    [UgradU1, UgradU2] = compute_UgradU_batch(U, V, hx, hy);
    %
    UgradU1_flat = reshape(UgradU1, [size(UgradU1,1)*size(UgradU1,2),...
        size(UgradU1,3)]);
    UgradU2_flat = reshape(UgradU2, [size(UgradU2,1)*size(UgradU2,2),...
        size(UgradU2,3)]);
    rhs_grad = -[UgradU1_flat; UgradU2_flat];  % shape => same stacked as above
    
    % Another stokes solve
    [U2, V2, P2] = ...
        stokes_solver_vectorized(rhs_grad, DiE, BDiE, DiBt, Di, B, Bt, ...
        perS, SS, SSt, Nx, Ny, sU);


    szU2 = size(U2);  % e.g., [Nx-1, Ny, L]
    U2_2d = reshape(U2, [szU2(1), szU2(2)*szU2(3)]);       
    U2dx_2d = - (A1') * U2_2d;                           
    U2dx = reshape(U2dx_2d, szU2);                       
    
    % same idea for U2dy => diff(U2, along dimension=2)
    U2dy = diff(U2, 1, 2) ./ hy;
    U2dy1 = 2 * U2(:,1,:) ./ hy;      % left boundary
    U2dy2 = -2 * U2(:,end,:) ./ hy;   % right boundary

    szV2 = size(V2);  % e.g., [Nx, Ny-1, L]
    % Flatten V2 => shape (Nx, (Ny-1)*L)
    V2_2d = reshape(V2, [szV2(1), szV2(2)*szV2(3)]);
    V2dx_2d = - (A1') * V2_2d;  % or however you define the sign
    V2dx = reshape(V2dx_2d, szV2);
    
    % For V2dy, we first pad columns and then take diff along columns
    V_ext = zeros(szV2(1), szV2(2)+2, szV2(3));
    V_ext(:, 2:end-1, :) = V2;
    % => shape (Nx, (Ny-1)+2, L)
    V2dy = diff(V_ext, 1, 2) ./ hy;  

    a = 0.5 * (inner_batch(U2, U2) + inner_batch(V2, V2)) ...
        + dt*mu * ( inner_batch(U2dx, U2dx) + inner_batch(U2dy, U2dy) ....
        + inner_batch(V2dx, V2dx) + inner_batch(V2dy, V2dy) ) ...
        + dt*mu*0.5 * ( inner_batch(U2dy1, U2dy1) + inner_batch(U2dy2, U2dy2));

    b = ( inner_batch(U - U1, U2) + inner_batch(V - V1, V2) ) ...
        - dt * ( inner_batch(UgradU1, U1) + inner_batch(UgradU2, V1) );

    c = -0.5 * ( inner_batch(U1 - U, U1 - U) + inner_batch(V1 - V, V1 - V) );

    a_new = a .* (hx*hy) + theta;
    b_new = b .* (hx*hy);
    c_new = c .* (hx*hy) - (q.^2).*theta;   % q shape => (L,)

    disc = b_new.^2 - 4 .* a_new .* c_new;
    % Solve for q_new => shape (L,)
    q_new = (-b_new + sqrt(disc)) ./ (2 .* a_new);

    q_new_resh = reshape(q_new, [1,1,length(q_new)]);

    U_new = U1 + bsxfun(@times, q_new_resh, U2);
    V_new = V1 + bsxfun(@times, q_new_resh, V2);
    P_new = P1 + bsxfun(@times, q_new_resh, P2);

end