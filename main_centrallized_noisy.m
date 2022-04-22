clearvars;
load routing_graph_2.mat;

% rng(0);

dim_total = idx_actions(end);
local_dims = diff(idx_actions);
eval_obj = @(x)(eval_obj_routing(x, cost_coeffs, flow2edge_mat, pathc2agentc_mat, aug_flow_mat, aug_flow_vec));
global_obj = @(x)(mean(eval_obj(x)));

n_trails = 50;

% parameters for sigma=0.01:
eta = 2e-3;
u = 1e-3;
delta = 0.1;
T = 50000;

% parameters for sigma=0.025:
% eta = 5e-4;
% u = 1.5e-3;
% delta = 0.15;
% T = 50000;

sigma = 0.01;

x0 = zeros(dim_total, 1);
for ii = 1:n_agent
    x0(idx_actions(ii)+1:idx_actions(ii+1)) = 1 / (idx_actions(ii+1) - idx_actions(ii) + 1);
end

obj_vals = zeros(T, n_trails);

for p = 1:n_trails
    obj_vals_tmp = zeros(T, 1);
    x_cur = x0;

    for t = 0:T-1
        z = zeros(dim_total,1);
        for ii = 1:n_agent
            x_local_cur = x_cur(idx_actions(ii)+1:idx_actions(ii+1));
            z_tilde = randn(local_dims(ii), 1);
            z(idx_actions(ii)+1:idx_actions(ii+1)) = sample_z(x_local_cur, u, ...
                z_tilde, local_dims(ii) + 1);
        end
        f_val_p = global_obj(x_cur + u * z) + sum(randn(n_agent, 1)/n_agent) * sigma * f_opt;
        f_val_m = global_obj(x_cur - u * z) + sum(randn(n_agent, 1)/n_agent) * sigma * f_opt;

        diff = (f_val_p - f_val_m) / (2*u);

        grad_est = diff * z;

        for ii = 1:n_agent
            x_cur(idx_actions(ii)+1:idx_actions(ii+1)) = mirror_descent_KL(x_cur(idx_actions(ii)+1:idx_actions(ii+1)), ...
                grad_est(idx_actions(ii)+1:idx_actions(ii+1)), eta, delta, local_dims(ii) + 1);
        end

        obj_vals_tmp(t+1) = global_obj(x_cur);
    end

    obj_vals(:, p) = obj_vals_tmp;
    fprintf('|');
    
end
fprintf('\n');


obj_mean = mean(obj_vals, 2);
obj_std = std(obj_vals, 0, 2);

fname = sprintf('dist_routing_centralized_noisy_%0.3f.mat', sigma);
save(fname);