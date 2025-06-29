function r = inner_batch(U, V)
    r = sum(sum(U .* V, 1), 2);
    r = squeeze(r);
   % r = reshape(r, [1, L]);

end