function f_t = f(t, eps_alpha)
    alpha_t = cond_alpha(t, eps_alpha);  % calls cond_alpha
    f_t = -(1 - eps_alpha) / alpha_t;
end