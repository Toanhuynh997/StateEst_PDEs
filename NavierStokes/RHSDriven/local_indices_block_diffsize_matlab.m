function flat_inds = local_indices_block_diffsize_matlab(p, Nx, Ny, radius)
    % Return 1-based global indices of the neighborhood around global index p,
    % where the state is composed of 4 blocks of sizes:
    %   Block 1: Nx x Ny
    %   Block 2: Nx x (Ny - 1)
    %   Block 3: Nx x Ny
    %   Block 4: scalar (1 value)

    % Block sizes
    m1 = Nx * Ny;
    m2 = Nx * (Ny - 1);
    m3 = Nx * Ny;
    % m4 = 1;
    % block_sizes = [m1, m2, m3, m4];
    block_sizes = [m1, m2, m3];
    % Offsets (1-based indexing)
    % offsets = [1, 1 + m1, 1 + m1 + m2, 1 + m1 + m2 + m3];
    offsets = [1, 1 + m1, 1 + m1 + m2];
    % Determine which block contains index p
    for comp = 1:length(block_sizes)
        if p < offsets(comp) + block_sizes(comp)
            break;
        end
    end

    % Local index within the component block (1-based)
    q = p - offsets(comp) + 1;

    % Get local neighborhood
    if comp == 1 || comp == 3
        % Blocks 1 & 3: Nx × Ny
        base_patch = local_indices_matlab(q, Nx, Ny, radius);
    elseif comp == 2
        % Block 2: Nx × (Ny - 1)
        base_patch = local_indices_matlab(q, Nx, Ny - 1, radius);
    % else
    %     % Block 4: scalar
    %     base_patch = 1;
    end

    % Convert to global 1-based indices
    flat_inds = base_patch + offsets(comp) - 1;
end
