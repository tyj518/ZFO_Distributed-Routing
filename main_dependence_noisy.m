clearvars;
load routing_graph_2.mat;
load comm_network;

% rng(0);

dim_total = idx_actions(end);
local_dims = diff(idx_actions);
eval_obj = @(x)(eval_obj_routing(x, cost_coeffs, flow2edge_mat, pathc2agentc_mat, aug_flow_mat, aug_flow_vec));
global_obj = @(x)(mean(eval_obj(x)));

n_trails = 50;

% parameters for sigma=0.01:
% eta_s = 3e-3 * ones(n_networks, 1);
% u_s = 1e-3 * ones(n_networks, 1);
% delta_s = 0.1 * ones(n_networks, 1);
% T_s = 50000 * ones(n_networks, 1);

% parameters for sigma=0.025:
eta_s = 7.5e-4 * ones(n_networks, 1);
u_s = 1.5e-3 * ones(n_networks, 1);
delta_s = 0.15 * ones(n_networks, 1);
T_s = 50000 * ones(n_networks, 1);

sigma = 0.025;

x0 = zeros(dim_total, 1);
for ii = 1:n_agent
    x0(idx_actions(ii)+1:idx_actions(ii+1)) = 1 / (idx_actions(ii+1) - idx_actions(ii) + 1);
end

obj_vals = cell(n_networks, 1);
obj_mean = cell(n_networks, 1);
obj_std = cell(n_networks, 1);
for test_case = 1:n_networks
    obj_vals{test_case} = zeros(T_s(test_case), n_trails);
    obj_mean{test_case} = zeros(T_s(test_case), 1);
    obj_std{test_case} = zeros(T_s(test_case), 1);
end

parfor test_case = 1:n_networks
    for p = 1:n_trails
        obj_vals{test_case}(1:T_s(test_case), p) = ZFO_one_run(eval_obj, global_obj, x0, ...
            eta_s(test_case), u_s(test_case), delta_s(test_case), T_s(test_case), sigma * f_opt, ...
            comm_dist{test_case}, Bmax(test_case), idx_actions, local_dims, dim_total, n_agent, ...
            fn_dependence);
    end
    fprintf('\n');

    obj_mean{test_case} = mean(obj_vals{test_case}, 2);
    obj_std{test_case} = std(obj_vals{test_case}, 0, 2);
end

fname = sprintf('dist_routing_noisy_depn_%0.3f.mat', sigma);
save(fname);