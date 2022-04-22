function [costs] = eval_obj_routing(x, cost_coeffs, ...
    flow2edge_mat, pathc2agentc_mat, aug_flow_mat, aug_flow_vec)

x_aug = aug_flow_mat * x + aug_flow_vec;
traffic_edge = flow2edge_mat * x_aug;

costs_edge = cost_coeffs(:, 1) .* traffic_edge.^2 + cost_coeffs(:, 2) .* traffic_edge + cost_coeffs(:, 3);

costs_path = flow2edge_mat' * costs_edge;
costs = pathc2agentc_mat * (costs_path .* x_aug);
end