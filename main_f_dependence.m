clearvars;
load data.mat;

rng(1);

eval_obj = @(x)(eval_obj_distributed_routing(x, nr, Q, routes_coeffs, agent_route_list)/abs(f_opt));
global_obj = @(x)(mean(eval_obj(x)));

% known function dependence
% noiseless: eta = 3e-2, u = 2e-3, delta = 0.05, T = 1500
% sigma = 0.02: eta = 1.5e-2, u = 4e-3, delta = 0.1, T = 7500
% sigma = 0.05: eta = 6e-3, u = 6e-3, delta = 0.15, T = 7500
eta = 3e-2;
u = 2e-3;
delta = 0.05;
sigma = 0.00;
T = 1500;

n_trails = 100;

obj_vals = zeros(T, n_trails);

x0 = zeros(sum(nr)-n, 1);
ptr = 1;
for j = 1:n
    x0(ptr:ptr+nr(j)-2) = ones(nr(j)-1, 1) / nr(j);
    ptr = ptr + nr(j) - 1;
end

for p = 1:n_trails
    obj_vals_tmp = zeros(T, 1);
    D_table = zeros(n, Bmax + 1);
    x_cur = x0;
    z_rec = zeros(sum(nr)-n, Bmax+1);
    
    for t = 0:T-1
        z = zeros(sum(nr)-n,1);
        ptr = 1;
        for j = 1:n
            z(ptr:ptr+nr(j)-2) = sample_z(x_cur(ptr:ptr+nr(j)-2), nr(j), u);
            ptr = ptr + nr(j) - 1;
        end
        z_rec = [z, z_rec(:, 1:end-1)];
        f_vals_p = eval_obj(x_cur + u * z) + sigma * randn(n, 1);
        f_vals_m = eval_obj(x_cur - u * z) + sigma * randn(n, 1);

        D_table = [(f_vals_p - f_vals_m) / (2*u), D_table(:, 1:end-1)];

        grad_est = zeros(sum(nr)-n, 1);
        ptr = 1;
        for k = 1:n
            par_grad_k = zeros(nr(k)-1,1);
            par_z_rec_k = z_rec(ptr:ptr+nr(k)-2, :);
            for j = fn_dependence{k}
                tau_kj = t - dist_mat(k, j);
                if tau_kj >= 0
                    par_grad_k = par_grad_k ...
                        + D_table(j, dist_mat(k, j)+1) * par_z_rec_k(:, dist_mat(k, j)+1);
                end
            end
            grad_est(ptr:ptr+nr(k)-2) = par_grad_k / n;
            
            ptr = ptr + nr(k) - 1;
        end

        ptr = 1;
        x_next_pre = x_cur - eta * grad_est;
        for k = 1:n
            x_cur(ptr:ptr+nr(k)-2) = proj_simplex_delta(x_next_pre(ptr:ptr+nr(k)-2), nr(k), delta);
            ptr = ptr + nr(k) - 1;
        end
        
        obj_vals_tmp(t+1) = global_obj(x_cur);
    end
    
    obj_vals(:, p) = obj_vals_tmp;
    fprintf('|');
end
fprintf('\n');

obj_mean = mean(obj_vals, 2);
obj_std = std(obj_vals, 0, 2);

fname = sprintf('dist_routing_sigma_%1.2f_depn_known.mat', sigma);
save(fname);