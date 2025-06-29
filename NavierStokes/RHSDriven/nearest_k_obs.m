function idx = nearest_k_obs(i, grid, obs_xy, scalar_idx, K)
    % Return the indices of the K nearest observations (from obs_xy) to grid point i
    % grid      : (N x 2) array of grid coordinates
    % obs_xy    : (M x 2) array of observation coordinates
    % scalar_idx: scalar index used for special case
    % K         : number of nearest neighbors to return

    % if i == scalar_idx
    %     idx = scalar_idx;
    %     return;
    % end

    % Compute squared Euclidean distances from grid(i,:) to all obs_xy
    diff = obs_xy - grid(i, :);  % M x 2
    d2 = sum(diff.^2, 2);        % M x 1

    % Find indices of K smallest distances
    [~, sorted_idx] = sort(d2);
    K_actual = min(K, numel(sorted_idx));
    idx = sorted_idx(1:K_actual);
end
