function [fvals] = eval_obj_distributed_routing(vs, nr, Q, routes_coeffs, agent_route_list)
n = length(Q);
[m, ~] = size(routes_coeffs);
vs_aug = cell(n, 1);

x_cr = zeros(m, 1);
ptr = 1;
for j = 1:n
    vj = vs(ptr:ptr+nr(j)-2);
    vs_aug{j} = [vj; 1-sum(vj)];
    x_cr(agent_route_list{j}) = x_cr(agent_route_list{j}) + ...
        Q(j) * vs_aug{j};
    ptr = ptr + nr(j) - 1;
end

x_cr_monomial = [x_cr.^2, x_cr, ones(m, 1)];
cr = sum(routes_coeffs .* x_cr_monomial, 2);

fvals = zeros(n, 1);
for j = 1:n
    fvals(j) = Q(j) * vs_aug{j}' * cr(agent_route_list{j});
end
end