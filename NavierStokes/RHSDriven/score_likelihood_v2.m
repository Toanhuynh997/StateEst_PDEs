function score =  score_likelihood_v2(xt, obs, t, C, obs_sigma)
    % # obs: (d)
    % # xt: (ensemble, d)
    score_x = -((xt) - obs)./(obs_sigma^2);
    tau = g_tau(t);
    score = tau.*score_x / C;
end