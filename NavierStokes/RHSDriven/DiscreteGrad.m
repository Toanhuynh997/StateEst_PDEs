function A = DiscreteGrad(N,h)
    % size(A) = [N,N], periodic
    A = spdiags(ones(N,1)*[-1 1],-1:0,N,N)/h;
    A(1,end) = -1/h;
end