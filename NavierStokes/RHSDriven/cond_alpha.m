function alpha_t = cond_alpha(t, eps_alpha)
    alpha_t = 1 - (1 - eps_alpha) * t;
end