function B = avg_batch(A)

    A = double(A);

    nd = ndims(A);
    sz = size(A);
    % 1D case
    if nd == 2 && (sz(1)==1 || sz(2)==1)
        % isvector in MATLAB
        % Flatten to 1D => average consecutive elements
        Aflat = A(:);  % shape = (n,1)
        B = 0.5 * (Aflat(2:end) + Aflat(1:end-1));
    % 2D case
    elseif nd == 2
        % shape (m, n)
        m = sz(1);
        % n = sz(2);
        if m == 1
            % Flatten row vector to 1D => average
            flat = A(:);
            B = 0.5 * (flat(2:end) + flat(1:end-1));
        else
            % Average consecutive rows => (m-1,n)
            B = 0.5 * (A(2:end,:) + A(1:end-1,:));
        end

    % 3) 3D case
    elseif nd == 3
        % shape (m, n, l)
        m = sz(1);
        % n = sz(2);
        % l = sz(3);
        if m == 1
            % Flatten everything => 1D => average consecutive
            flat = A(:);
            B = 0.5 * (flat(2:end) + flat(1:end-1));
        else
            % Average along dimension 1 => shape (m-1, n, l)
            B = 0.5 * (A(2:end,:,:) + A(1:end-1,:,:));
        end
    else
        error('avg_batch is only implemented for up to 3D arrays.');
    end
end

