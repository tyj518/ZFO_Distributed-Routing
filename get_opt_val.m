function [x_opt, f_opt] = get_opt_val(idx_actions, cost_coeffs, ...
    flow2edge_mat, aug_flow_mat, aug_flow_vec)

fun = @(x)(fun_fmincon(x, ...
    cost_coeffs, flow2edge_mat, aug_flow_mat, aug_flow_vec));

n_agent = length(idx_actions) - 1;
x0 = zeros(idx_actions(end), 1);
A_mat = zeros(n_agent, idx_actions(end));
for ii = 1:n_agent
    x0(idx_actions(ii)+1:idx_actions(ii+1)) = 1 / (idx_actions(ii+1)-idx_actions(ii)+1);
    A_mat(ii, idx_actions(ii)+1:idx_actions(ii+1)) = 1;
end

options = optimoptions('fmincon','SpecifyObjectiveGradient',true);
[x_opt, f_opt] = fmincon(fun, x0, A_mat, ones(n_agent, 1), [], [], zeros(idx_actions(end), 1), [], [], options);
end

function [fval, grad] = fun_fmincon(x, cost_coeffs, flow2edge_mat, aug_flow_mat, aug_flow_vec)

x_aug = aug_flow_mat * x + aug_flow_vec;
traffic_edge = flow2edge_mat * x_aug;
fval = sum(cost_coeffs(:, 1) .* traffic_edge.^3 + cost_coeffs(:, 2) .* traffic_edge.^2 + cost_coeffs(:, 3) .* traffic_edge);

tmp_grad = 3 * cost_coeffs(:, 1) .* traffic_edge.^2 + 2 * cost_coeffs(:, 2) .* traffic_edge + cost_coeffs(:, 3);
grad = aug_flow_mat' * flow2edge_mat' * tmp_grad;

end