function g2 = g_sq(t, eps_alpha)
    d_sigma_sq_dt = 1;
    val_f = f(t, eps_alpha);        % calls f
    sigma2_t = cond_sigma_sq(t);    % calls cond_sigma_sq

    g2 = d_sigma_sq_dt - 2 * val_f * sigma2_t;
end