function [U_new, V_new, P_new, q_new] = NS_BDF2_1step_vectorize(...
    hx, hy, dt, U_old, U_old_old, V_old, V_old_old, q_old, q_old_old,...
    mu, theta, DiE, BDiE, DiBt, Di, B, Bt, perS, SS, SSt, Nx, Ny, sU,...
    alpha, A1, ff1, ff2)

    rhs1 = ff1+(4*U_old-U_old_old)/(2*dt);   
    rhs2 = ff2+(4*V_old-V_old_old)/(2*dt);

    rhs1_flat = reshape(rhs1, [(size(rhs1,1)*size(rhs1,2)), ...
        size(rhs1,3)]);  % MATLAB is col-major
    rhs2_flat = reshape(rhs2, [(size(rhs2,1)*size(rhs2,2)),...
        size(rhs2,3)]);
    
    rhs_full = [rhs1_flat; rhs2_flat];  % shape => ( (Nx-1)*Ny + Nx*(Ny-1), L )
    
    % Solve stokes system (batch)
    [U1, V1, P1] = ...
        stokes_solver_vectorized(rhs_full, DiE, BDiE, DiBt, Di, B, Bt, ...
                                 perS, SS, SSt, Nx, Ny, sU);


    [UgradU1, UgradU2] = ...
        compute_UgradU_batch(2*U_old-U_old_old,2*V_old-V_old_old, hx, hy);
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

    a = 1.5 * (inner_batch(U2, U2) + inner_batch(V2, V2)) ...
        + 2*dt*mu * ( inner_batch(U2dx, U2dx) + inner_batch(U2dy, U2dy) ....
        + inner_batch(V2dx, V2dx) + inner_batch(V2dy, V2dy) ) ...
        + 2*dt*mu*0.5 * ( inner_batch(U2dy1, U2dy1) + inner_batch(U2dy2, U2dy2));

    b = ...
        -(inner_batch(3*U1-4*U_old+U_old_old, U2)...
        -inner_batch(3*V1-4*V_old+V_old_old, V2)) ...
        - 2*dt*( inner_batch(UgradU1, U1) + inner_batch(UgradU2, V1) );

    c = -2*(inner_batch(U1-U_old,U1-U_old)+inner_batch(V1-V_old,V1-V_old))...
        +0.5*(inner_batch(U1-U_old_old,U1-U_old_old)+...
        inner_batch(V1-V_old_old,V1-V_old_old));

    a_new = a .* (hx*hy) + 3*theta;
    b_new = b .* (hx*hy);
    c_new = c .* (hx*hy) - theta*(4*q_old.^2-q_old_old.^2);;   % q shape => (L,)

    disc = b_new.^2 - 4 .* a_new .* c_new;
    if any(disc<0)
        idx = find(disc<0);
        warning('Warning: q is complex at entries %s', mat2str(idx));
    end
    
    sqrtDelta     = sqrt(disc);                    % works element-wise (complex if delta<0)
    q1        = (-b + sqrtDelta) ./ (2.*a);
    q2        = (-b - sqrtDelta) ./ (2.*a);
    
    % choose the root that’s closer to 1, element-wise
    mask      = abs(q1 - 1) < abs(q2 - 1);      
    q_new         = q2;                             % pre-allocate with the "other" root
    q_new(mask)   = q1(mask);  
 
    q_new_resh = reshape(q_new, [1,1,length(q_new)]);

    U_new = U1 + bsxfun(@times, q_new_resh, U2);
    V_new = V1 + bsxfun(@times, q_new_resh, V2);
    P_new = P1 + bsxfun(@times, q_new_resh, P2);

end