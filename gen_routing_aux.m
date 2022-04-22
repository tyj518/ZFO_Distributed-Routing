clearvars;

load raw_routing_graph_2.mat;

idx_actions = zeros(n_agent+1, 1);
for ii = 1:n_agent
    idx_actions(ii+1) = idx_actions(ii) + length(paths{ii, 2})-1;
end
local_dims = diff(idx_actions);

flow2edge_mat = zeros(n_edge, idx_actions(end) + n_agent);
pathc2agentc_mat = zeros(n_agent, idx_actions(end) + n_agent);
aug_flow_mat = zeros(idx_actions(end) + n_agent, idx_actions(end));
aug_flow_vec = zeros(idx_actions(end) + n_agent, 1);
ptr = 1;
for ii = 1:n_agent
    n_paths = length(paths{ii, 2});
    for pp = 1:n_paths
        flow2edge_mat(paths{ii,2}{pp}, ptr) = flow2edge_mat(paths{ii,2}{pp}, ptr) + 1;
        pathc2agentc_mat(ii, ptr) = pathc2agentc_mat(ii, ptr) + 1;
        ptr = ptr + 1;
    end
    aug_flow_mat(idx_actions(ii)+ii:idx_actions(ii+1)+ii-1, idx_actions(ii)+1:idx_actions(ii+1)) = traffic_agent(ii) * eye(local_dims(ii));
    aug_flow_mat(idx_actions(ii+1)+ii, idx_actions(ii)+1:idx_actions(ii+1)) = -traffic_agent(ii);
    aug_flow_vec(idx_actions(ii+1)+ii) = traffic_agent(ii);
end
flow2edge_mat = sparse(flow2edge_mat);
pathc2agentc_mat = sparse(pathc2agentc_mat);
aug_flow_mat = sparse(aug_flow_mat);

fn_dependence = cell(n_agent, 1);
for ii = 1:n_agent
    idx_edges = find(sum(flow2edge_mat(:, idx_actions(ii)+ii:idx_actions(ii+1)+ii), 2));
    fn_dependence{ii} = find(pathc2agentc_mat * sum(flow2edge_mat(idx_edges, :), 1)' > 0);
end

[x_opt, f_opt] = get_opt_val(idx_actions, cost_coeffs, flow2edge_mat, aug_flow_mat, aug_flow_vec);
f_opt = f_opt / n_agent;

save routing_graph_2.mat traffic_agent cost_coeffs flow2edge_mat pathc2agentc_mat aug_flow_mat aug_flow_vec n_edge n_agent idx_actions local_dims x_opt f_opt fn_dependence;