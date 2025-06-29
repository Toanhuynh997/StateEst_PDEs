function [U_out, V_out, P_out] = ...
    stokes_solver_vectorized(f, DiE, BDiE, DiBt, Di, B, Bt, perS,...
    SS, SSt, Nx, Ny, sU)

    f = double(f);

    nd = ndims(f);
    szf = size(f);
    if isvector(f)
        [U_out, V_out, P_out] = stokes_solver(f(:),...
            DiE, BDiE, DiBt, Di, B, Bt, perS, SS, SSt, Nx, Ny, sU);

    elseif nd == 2
        % n = szf(1);
        L = szf(2);
        rhs_1 = Di * f;        % shape (n, L)
        rhs_2 = B * rhs_1;     % shape (n, L)
        rhs_2(perS, :) = SS\(SSt\rhs_2(perS, :));

        % perS_mask = false(n,1);
        % perS_mask(perS) = true;
        % 
        % rhs_2_sub = rhs_2(perS_mask, :);
        % temp = SSt \ rhs_2_sub;        % shape (len(perS), L)
        % rhs_2(perS_mask,:) = SS \ temp;    % shape (len(perS), L)
    
        rhs = rhs_1 - Di * (Bt * rhs_2);   % shape (n, L)

        % Prepare storage
        U_list = cell(L,1);
        V_list = cell(L,1);
        P_list = cell(L,1);
    
        % GMRES parameters
        % tol = 1e-8;
        % maxit = 1000;
        all_sol = zeros(size(rhs, 1), L);
        for i = 1:L
            rhs_col = rhs(:, i);
            % opts = struct('Display', 'none');
            lhs = @(u) lhs_u(u,DiE,BDiE,DiBt,perS,SS,SSt);

            [sol_i, flag] = gmres(lhs,rhs_col,[],1e-8,1000,[],[],[]);
            % if flag_i ~= 0
            %     warning('GMRES did not converge for batch=%d, info=%d', i, flag_i);
            % end
            all_sol(:, i) = sol_i;
            % Extract velocity components
            U_i = sol_i(1:sU);
            V_i = sol_i(sU+1:end);
    
            % Reshape
            % U_i => (Nx, Ny) by column-major
            U_i = reshape(U_i, [Nx, Ny]);
            V_i = reshape(V_i, [Nx, Ny-1]);
            
            U_list{i} = U_i;
            V_list{i} = V_i;

        end
        % Post-processing
        p = BDiE*all_sol;
        p(perS, :) = SS\(SSt\p(perS, :));
        p = p+rhs_2;

        % p = BDiE*sol_i;
        % p(perS) = SS\(SSt\p(perS));
        % p = p+rhs_2(:, i);
          
        for i = 1:L
            % p_ex
            p_ex = [0; p(:, i)];    % length n+1
            % remove mean
            p_ex = p_ex - mean(p_ex);
            P_i = reshape(p_ex, [Nx, Ny]);
          
            P_list{i} = P_i;
        end

        % Stack along 3rd dim
        U_out = cat(3, U_list{:});  % shape => (Nx, Ny, L)
        V_out = cat(3, V_list{:});
        P_out = cat(3, P_list{:});
    end
end

function outvec = lhs_op(invec, DiE, BDiE, DiBt, perS, SS, SSt)
    outvec = lhs_u(invec, DiE, BDiE, DiBt, perS, SS, SSt);
end
    