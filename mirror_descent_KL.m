function [x] = mirror_descent_KL(x0, G, eta, delta, n_dim)

x0_aug = [x0; 1-sum(x0)];
G_aug = [G; 0];

tmp = x0_aug .* exp(-eta * G_aug);
[tmp_srt, idx_srt] = sort(tmp);
cumsum_tmp = cumsum(tmp_srt, 'reverse');

c = (1 - (0:n_dim-1)' * delta/n_dim) ./ cumsum_tmp .* tmp_srt;

idx_th = find(c >= delta/n_dim, 1);

if isempty(idx_th)
    idx_th = n_dim;
end

x = delta/n_dim * ones(n_dim, 1);
x(idx_srt(idx_th:end)) = (1 - (idx_th-1) * delta/n_dim) / cumsum_tmp(idx_th) * tmp_srt(idx_th:end);
x = x(1:end-1);
end