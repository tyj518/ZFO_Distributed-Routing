function [fval, grad] = global_func_opt(x, Q, nr, routes_coeffs, agent_route_list)
n = length(Q);
m = size(routes_coeffs, 1);
x_cr = zeros(m, 1);
ptr = 1;
for j = 1:n
    vs_aug = x(ptr:ptr+nr(j)-1);
    x_cr(agent_route_list{j}) = x_cr(agent_route_list{j}) + ...
        Q(j) * vs_aug;
    ptr = ptr + nr(j);
end
fval = sum(sum(routes_coeffs .* [x_cr.^3, x_cr.^2,x_cr])) / n;

grad = zeros(length(x), 1);
ptr = 1;
for j = 1:n
    grad(ptr:ptr+nr(j)-1) = grad(ptr:ptr+nr(j)-1) + ...
        Q(j) * sum(routes_coeffs(agent_route_list{j}, :) ...
        .* [3 * x_cr(agent_route_list{j}).^2, 2 * x_cr(agent_route_list{j}), ones(nr(j), 1)], 2);
    ptr = ptr + nr(j);
end
grad = grad / n;
end