function flat_inds = local_indices_matlab(p, Nx, Ny, r)
    % Return the flattened indices of a square neighborhood of radius r
    % around grid point p on an Nx-by-Ny grid (1-based indexing).
    %
    % p  : scalar index in [1, Nx*Ny]
    % Nx : number of rows
    % Ny : number of columns
    % r  : radius of neighborhood (in grid cells)
    %
    % Output: flat_inds — column vector of linear indices (1-based)
    p0 = p-1;
    % Convert flat index p -> (i, j) using 1-based indexing
    i = floor(p0 / Ny) + 1;
    j = mod(p0, Ny) + 1;

    % Define row/col ranges, clipped to grid size
    row_min = max(0, i - r);
    row_max = min(Nx-1, i + r);
    col_min = max(0, j - r);
    col_max = min(Ny-1, j + r);

    rows = row_min:row_max;
    cols = col_min:col_max;

    % Form Cartesian product (grid)
    [Ii, Jj] = ndgrid(rows, cols);

    % Flattened 1-based linear indices
    flat_inds0 = Ii*Ny+Jj;
    flat_inds = flat_inds0(:)+1;
end
