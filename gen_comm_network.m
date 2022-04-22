clearvars;
load routing_graph_2.mat n_agent idx_actions;

rng(1);

n_networks = 3;

comm_dist = cell(n_networks, 1);
b_bar = zeros(n_networks, 1);
Bmax = zeros(n_networks, 1);

% grid network
n_row = 4;
n_col = floor(n_agent / n_row);
n_res = n_agent - n_row * n_col;
adj = zeros(n_agent, n_agent);
for rr = 1:n_row
    for cc = 1:n_col
        cur_idx = (rr-1) * n_col + cc;
        if rr > 1
            up_node_idx = (rr-2) * n_col + cc;
            adj(cur_idx, up_node_idx) = 1;
            adj(up_node_idx, cur_idx) = 1;
        end
        if rr < n_row
            down_node_idx = rr * n_col + cc;
            adj(cur_idx, down_node_idx) = 1;
            adj(down_node_idx, cur_idx) = 1;
        end
        if cc > 1
            left_node_idx = (rr-1) * n_col + cc - 1;
            adj(cur_idx, left_node_idx) = 1;
            adj(left_node_idx, cur_idx) = 1;
        end
        if cc < n_col
            right_node_idx = (rr-1) * n_col + cc + 1;
            adj(cur_idx, right_node_idx) = 1;
            adj(right_node_idx, cur_idx) = 1;
        end
    end
end
for ii = 1:n_res
    cur_idx = n_row * n_col + ii;
    up_node_idx = (n_row-1) * n_col + ii;
	adj(cur_idx, up_node_idx) = 1;
	adj(up_node_idx, cur_idx) = 1;
    
    if ii > 1
        left_node_idx = n_row * n_col + ii - 1;
        adj(cur_idx, left_node_idx) = 1;
        adj(left_node_idx, cur_idx) = 1;
    end
    
    if ii < n_res
        right_node_idx = n_row * n_col + ii + 1;
        adj(cur_idx, right_node_idx) = 1;
        adj(right_node_idx, cur_idx) = 1;
    end
end
G_comm = graph(adj);
comm_dist{1} = distances(G_comm);


% Erdos-Renyi network
prob = 0.05;
Fiedler = 0;    % Fiedler eigenvalue, strictly positive if connected
while Fiedler < 1e-6
    adj = binornd(1, prob, n_agent, n_agent);
    adj = tril(adj) + triu(adj', 1);
    adj = adj - diag(diag(adj));

    degs = sum(adj, 2);
    laplacian = diag(degs) - adj;

    Fiedler = max(mink(eig(laplacian), 2));
end
G_comm = graph(adj);
comm_dist{2} = distances(G_comm);

% linear network
adj = diag(ones(n_agent-1, 1), 1) + diag(ones(n_agent-1, 1), 1)';
G_comm = graph(adj);
comm_dist{3} = distances(G_comm);


local_dims = diff(idx_actions);
for kk = 1:n_networks
    Bmax(kk) = max(max(comm_dist{kk}));
    b_bar(kk) = sqrt(sum(sum(comm_dist{kk}.^2 .* repmat(local_dims, 1, n_agent))) / (n_agent * sum(local_dims)));
end

save comm_network.mat comm_dist Bmax b_bar n_networks;