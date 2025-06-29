function A = DiscreteLaplace(N,h)
    % periodic
    A = spdiags(ones(N,1)*[1 -2 1],-1:1,N,N)/h^2;
    A(1,end) = 1/h^2; A(end,1) = 1/h^2;
end