function [UgradU1, UgradU2] = compute_UgradU_batch(U, V, hx, hy)

    % Ensure numeric
    U = double(U);
    V = double(V);
    
    %%
    Ucat = cat(1, U(end,:,:), U, U(1,:,:));  % shape => (M+2, N, L)
    Udx  = (Ucat(3:end,:,:) - Ucat(1:end-2,:,:)) / (2*hx);  % shape => (M, N, L)

    [M, N, L] = size(U);
    Udy = zeros(M, N, L);
    % Middle columns (2..N-1 in MATLAB)
    if N > 2
        Udy(:,2:end-1,:) = (U(:,3:end,:) - U(:,1:end-2,:)) / (2*hy);
    end
    % First column
    Udy(:,1,:) = (U(:,2,:) + U(:,1,:)) / (2*hy);
    % Last column
    Udy(:,end,:) = -(U(:,end,:) + U(:,end-1,:)) / (2*hy);

    Vmod = zeros(size(V,1), size(V,2)+2, size(V,3));  % => shape (N, M+2, L)
    Vmod(:,2:end-1,:) = V;

    Vmod = cat(1, Vmod(end,:,:), Vmod);

    Vmod = avg_batch(Vmod);

    Vmod_t = permute(Vmod, [2, 1, 3]);  % => shape => (M+2, N, L)
    Vmod_t = avg_batch(Vmod_t);            % => shape => (M+1, N, L)
    Vmod   = permute(Vmod_t, [2, 1, 3]);% => shape => (N, M+1, L)
    UgradU1 = U .* Udx + Vmod .* Udy; 
    
    %%
    V_ext = cat(1, V(end,:,:), V, V(1,:,:));
    Vdx = (V_ext(3:end,:,:) - V_ext(1:end-2,:,:)) ./ (2*hx);

    col0 = zeros(N,1,L);
    left  = cat(2, col0, V(:,1:end-1,:));   % shape => (N,M,L)
    right = cat(2, V(:,2:end,:), col0);     % shape => (N,M,L)
    Vdy = (right - left) ./ (2*hy);

    U_ext2 = cat(1, U, U(1,:,:));  % => (M+1, N, L)
    tmp_t  = permute(U_ext2, [2,1,3]);
    tmp_t2 = avg_batch(tmp_t);            % shape => (N, M, L)
    U_ext2 = permute(tmp_t2, [2,1,3]);    % shape => (M, N, L)
    Umod   = avg_batch(U_ext2); 
    UgradU2 = Umod .* Vdx + V .* Vdy;
end