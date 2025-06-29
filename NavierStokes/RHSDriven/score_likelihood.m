function score =  score_likelihood(xt, obs, t, C, obs_sigma)
    % # obs: (d)
    % # xt: (ensemble, d)
    % score_x = -(atan(xt) - obs)./(obs_sigma^2) .* (1./(1+xt.^2));
    score_x = -(atan(xt) - obs)./(obs_sigma^2) .* (1./(1+xt.^2));
    tau = g_tau(t);
    score = tau.*score_x / C;
end