function [z] = sample_z(v, n_r, u)
z0 = randn(n_r-1, 1);

sum_gap = (1 - sum(v)) / u;
sum_z0 = sum(z0);

if min(v + u*z0) >= 0 && min(v - u*z0) >= 0 ...
        && sum_z0 <= sum_gap && sum_z0 >= -sum_gap
    z = z0;
else
    % fprintf('z truncated. ');
    
    eps = 1e-9;
    rho = 2;
    
    y_old = 1/n_r*ones(n_r-1,1);
    
    z = max(min((z0+rho*y_old)/(1+rho), v/u), -v/u);
    y = z;
    sum_y_tmp = sum(y);
    if sum_y_tmp > sum_gap
        y = y + (sum_gap - sum_y_tmp)/(n_r-1);
    elseif sum_y_tmp < -sum_gap
        y = y + (-sum_gap - sum_y_tmp)/(n_r-1);
    end
    lambda = rho * (z-y);
%     k = 1;
    
    res1 = norm(z-y);
    res2 = rho*norm(y-y_old);
    while res1 >= eps || res2 >= eps
        
%         if res1 > 5*res2
%             rho = rho * 2;
%         elseif res2 > 5*res1
%             rho = rho/2;
%         end
        
        y_old = y;
        z = max(min((z0+rho*y-lambda)/(1+rho), v/u), -v/u);
        y = z+lambda/rho;
        sum_y_tmp = sum(z+lambda/rho);
        if sum_y_tmp > sum_gap
            y = y + (sum_gap - sum_y_tmp)/(n_r-1);
        elseif sum_y_tmp < -sum_gap
            y = y + (-sum_gap - sum_y_tmp)/(n_r-1);
        end
        lambda = lambda + rho * (z-y);
%         k = k+1;
        
        res1 = norm(z-y);
        res2 = rho*norm(y-y_old);
    end
%     disp(k);
    
    r_z = abs(sum(z)) / sum_gap;
    if r_z > 1
        z = z / r_z;
    end
    
%     z_test = quadprog(eye(n_r-1), -z0, [ones(1,n_r-1); -ones(1,n_r-1)], [sum_gap;sum_gap], [],[], -v/u, v/u);
%     norm(z-z_test)
end
end