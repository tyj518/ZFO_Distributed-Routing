function [obj_vals] = ZFO_one_run(eval_obj, global_obj, x0, eta, u, delta, T, sigma, ...
    dist_mat, Bmax, idx_actions, local_dims, dim_total, n_agent, fn_dependence)

obj_vals = zeros(T, 1);
D_table = zeros(n_agent, Bmax + 1);
x_cur = x0;
z_rec = zeros(dim_total, Bmax + 1);

for t = 0:T-1
    z = zeros(dim_total,1);
    for ii = 1:n_agent
        x_local_cur = x_cur(idx_actions(ii)+1:idx_actions(ii+1));
        z_tilde = randn(local_dims(ii), 1);
        z(idx_actions(ii)+1:idx_actions(ii+1)) = sample_z(x_local_cur, u, ...
            z_tilde, local_dims(ii) + 1);
    end
    z_rec = [z, z_rec(:, 1:end-1)];
    f_vals_p = eval_obj(x_cur + u * z) + sigma * randn(n_agent, 1);
    f_vals_m = eval_obj(x_cur - u * z) + sigma * randn(n_agent, 1);

    D_table = [(f_vals_p - f_vals_m) / (2*u), D_table(:, 1:end-1)];

    grad_est = zeros(dim_total, 1);
    
    if nargin == 15     % known local function dependence
        for ii = 1:n_agent
            par_grad_ii = zeros(local_dims(ii),1);
            par_z_rec_ii = z_rec(idx_actions(ii)+1:idx_actions(ii+1), :);
            for jj = fn_dependence{ii}'
                tau_ij = t - dist_mat(ii, jj);
                if tau_ij >= 0
                    par_grad_ii = par_grad_ii ...
                        + D_table(jj, dist_mat(ii, jj)+1) * par_z_rec_ii(:, dist_mat(ii, jj)+1);
                end
            end
            grad_est(idx_actions(ii)+1:idx_actions(ii+1)) = par_grad_ii / n_agent;
        end
    else                % unknown local function dependence
        for ii = 1:n_agent
            par_grad_ii = zeros(local_dims(ii),1);
            par_z_rec_ii = z_rec(idx_actions(ii)+1:idx_actions(ii+1), :);
            for jj = 1:n_agent
                tau_ij = t - dist_mat(ii, jj);
                if tau_ij >= 0
                    par_grad_ii = par_grad_ii ...
                        + D_table(jj, dist_mat(ii, jj)+1) * par_z_rec_ii(:, dist_mat(ii, jj)+1);
                end
            end
            grad_est(idx_actions(ii)+1:idx_actions(ii+1)) = par_grad_ii / n_agent;
        end
    end

    for ii = 1:n_agent
        x_cur(idx_actions(ii)+1:idx_actions(ii+1)) = mirror_descent_KL(x_cur(idx_actions(ii)+1:idx_actions(ii+1)), ...
            grad_est(idx_actions(ii)+1:idx_actions(ii+1)), eta, delta, local_dims(ii) + 1);
    end

    obj_vals(t+1) = global_obj(x_cur);
end
fprintf('|');
end