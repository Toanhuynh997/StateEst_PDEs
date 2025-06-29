function flat_inds = local_indices_block_matlab(p, Nx, Ny, radius)
    % Return 1-based global indices of a radius-'radius' neighborhood
    % around global index p in a 3-component state [X1; X2; X3], each of size (Nx * Ny).
    %
    % p      : scalar in [1, 3*Nx*Ny]
    % Nx, Ny : grid dimensions
    % radius : neighborhood radius

    m = Nx * Ny;

    % Determine which component (1, 2, or 3)
    comp = floor((p - 1) / m);  % 0-based component index (0, 1, or 2)

    % Local index within component block (1-based)
    q = p - comp * m;

    % Local neighborhood within this component
    base_patch = local_indices_matlab(q, Nx, Ny, radius);  % 1-based output

    % Shift indices to global index
    flat_inds = base_patch + comp * m;  % Still 1-based
end
