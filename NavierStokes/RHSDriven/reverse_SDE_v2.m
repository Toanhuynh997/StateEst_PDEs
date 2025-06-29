function [xt, path_all, t_vec] = reverse_SDE_v2(x0, obs, time_steps, C,...
    score_id, save_path, eps_alpha, obs_sigma)
    
    dt = 1.0 / time_steps;

    % Initialize xt from standard Gaussian (like x_T = randn in Python)
    xt = randn(size(x0));  % same size as x0
    
    t = 1.0;  % start from t=1.0 going backward

    % Prepare arrays for path
    if save_path
        path_all = cell(time_steps+1, 1);
        path_all{1} = xt;
        t_vec = zeros(time_steps+1,1);
        t_vec(1) = t;
    else
        path_all = {};
        t_vec = [];
    end

    % -- Step 2: Forward Euler (reverse-time) iterations
    for i = 1:time_steps

        alpha_t = cond_alpha(t, eps_alpha);
        sigma2_t = cond_sigma_sq(t);
        diffuse = g(t, eps_alpha);
        if score_id == 1
            score_val = score_likelihood_v2(xt, obs, t, C, obs_sigma);
            xt = ...
                xt-dt*(f(t, eps_alpha)*xt+diffuse^2*((xt-alpha_t*x0)/sigma2_t)...
                -diffuse^2*score_val)+sqrt(dt)*diffuse.*randn(size(xt));
        else
            xt = xt-dt*(f(t, eps_alpha)*xt+diffuse^2*((xt-alpha_t*x0)/sigma2_t) ) ...
                 + sqrt(dt)*diffuse.*randn(size(xt));
        end

        % Save path if requested
        if save_path
            path_all{i+1} = xt;
            t_vec(i+1) = t;
        end

        % Update time
        t = t - dt;
    end

    % If not saving path, return empty
    if ~save_path
        path_all = {};
        t_vec = [];
    end

end