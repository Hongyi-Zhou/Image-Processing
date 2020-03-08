function [x_hat] = IHT(y, K, psi, psi_adj, A, A_adj, max_iter, x_init)
%ITERATIVE_HARD_THRESHOLDING 
% return x_hat = argmin_x  0.5 ||y - A(x)||^2_2 s.t. || phi(x) ||_0 <= K

if ~exist('x_init', 'var')
    x_hat = A_adj(y);
else
    x_hat = x_init;
end

alpha = 0.01;

for iter = 1:max_iter
    % This is basically a fixed point method (or projected gradient descent)
    % first we perform the typical gradient descent for the first part
    gradient_x = -A_adj(y-A(x_hat));
    x_hat = x_hat - alpha * gradient_x; 
    
    % Then project x into region that satisfies ||phi(x)||_0 <= K
    psi_xhat = psi(x_hat);
    [~,idx] = sort(abs(psi_xhat), 'descend');
    psi_xhat(idx(K+1:end)) = 0;
    x_hat = psi_adj(psi_xhat);
    
    obj = 0.5 * norm(vec(y-A(x_hat)))^2;
    fprintf('iter = %d obj = %f\n', iter, obj);
end


end

function y = vec(x)
y = x(:);
end

