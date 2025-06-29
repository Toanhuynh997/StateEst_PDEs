function Xa = LETKF_local_matlab(Xb, y, R, Nx, Ny, idx_obs, ...
    grid_xy_total, obs_xy, radius, rho)
    % Local LETKF using a square neighborhood (no observation localization)
    %
    % Inputs:
    %   Xb              - [m_g x k] forecast ensemble
    %   y               - [l_g x 1] observation vector
    %   R               - [l_g x l_g] observation error covariance
    %   Nx, Ny          - grid dimensions
    %   idx_obs         - observation indices (1-based)
    %   grid_xy_total   - [m_g x 2] grid coordinates for each state variable
    %   obs_xy          - [l_g x 2] coordinates of observations
    %   radius          - localization radius
    %   rho             - inflation factor (you passed this in original M definition)
    %
    % Output:
    %   Xa              - [m_g x k] analysis ensemble

    [m_g, k] = size(Xb);

    % Step 1 & 2: global means & perturbations
    Xb_mean = mean(Xb, 2);                   % [m_g x 1]
    Xb_pert = Xb - Xb_mean;                  % [m_g x k]
    
    Yb_raw  = atan(Xb(idx_obs, :));          % [l_g x k]
    Yb_mean = mean(Yb_raw, 2);               % [l_g x 1]
    Yb_pert = Yb_raw - Yb_mean;              % [l_g x k]

    % Initialize output
    Xa = zeros(size(Xb));

    for p = 1:m_g
    % for p = (m_g-1):m_g
        % disp(p)
        % Step 3: select local model indices (1-based)
        loc_inds = local_indices_block_diffsize_matlab(p, Nx, Ny, radius);  % column vector

        xb_m_loc = Xb_mean(loc_inds);         % [m_loc x 1]
        Xb_loc   = Xb_pert(loc_inds, :);      % [m_loc x k]

        % Step 4: use all observations (nearest only, no localization weights)
        loc_obs = nearest_k_obs(p, grid_xy_total, obs_xy, m_g, 15);  % 1-based

        yb_m_loc = Yb_mean(loc_obs);          % [l_loc x 1]
        Yb_loc   = Yb_pert(loc_obs, :);       % [l_loc x k]
        yo_loc   = y(loc_obs);                % [l_loc x 1]

        R_loc = R(loc_obs, loc_obs);          % [l_loc x l_loc]

        % Step 5–6: analysis in ensemble space
        W = R_loc \ Yb_loc;                   % [l_loc x k]
        C = W';                               % [k x l_loc]

        M = (k - 1) / rho * eye(k) + C * Yb_loc;  % [k x k]
        M = 0.5*(M+M');
        [Q, D] = eig(M);                      % Q: eigenvectors, D: diag eigenvalues
        w = diag(D);
        w_inv = 1 ./ w;
        w_inv_sqrt = 1 ./ sqrt(w);

        Pa = Q * diag(w_inv) * Q';            % [k x k]
        Wa = sqrt(k - 1) * Q * diag(w_inv_sqrt) * Q';  % [k x k]

        wabar = Pa * (C * (yo_loc - yb_m_loc));     % [k x 1]
        Wana  = Wa + wabar;                         % [k x k]

        % Step 8: back to model space
        xa_loc = xb_m_loc + Xb_loc * Wana;          % [m_loc x k]

        % Identify center index in loc_inds
        Xa(p, :) = xa_loc(loc_inds == p, :);
    end
end
