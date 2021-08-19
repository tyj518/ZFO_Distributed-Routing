function [v] = proj_simplex_delta(v0, n_r, delta)

zero_vec = zeros(n_r-1,1);

v0_tmp = max(v0 - delta/n_r, zero_vec);
[v0_tmp_srt, perm] = sort(v0_tmp, 'descend');
cumsum_v0_srt = cumsum(v0_tmp_srt);
mu_tmp = max((cumsum_v0_srt - (1-delta))./[1:n_r-1]', zero_vec);
idx = find(v0_tmp_srt - mu_tmp >= 0, 1, 'last');
v = max(v0_tmp_srt - mu_tmp(idx), zero_vec) + delta/n_r;
v(perm) = v;


% v2 = quadprog(eye(n_r-1), -v0, ones(1,n_r-1), 1-delta/n_r, [], [], ones(n_r-1,1)*delta/n_r,[]);
% norm(v2-v)
end