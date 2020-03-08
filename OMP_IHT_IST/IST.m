function [x_hat] = IST(y, beta, psi, psi_adj, A, A_adj, max_iter, x_init)
%ITERATIVE_SOFT_THRESHOLDING 
% return x_hat = argmin_x  0.5 ||y - A(x)||^2_2 + beta || phi(x) ||_1 

if ~exist('x_init', 'var')
    x_hat = A_adj(y);
else
    x_hat = x_init;
end

alpha = 0.01;

for iter = 1:max_iter
    % this is a fixed point method (or projected gradient descent)
    %first we perform the typical gradient descent for the first part
    gradient_x = -A_adj(y-A(x_hat));
    x_hat = x_hat - alpha * gradient_x;
    
    % then we solve the problem min_x 0.5 || x - x_hat ||_2^2 + beta/alpha ||psi(x)||_1
    % let s = psi(x), we have min_s 0.5 || psi_adj(s) - x_hat ||_2^2 + beta/alpha ||s||_1
    % which has the same optimal as min_s 0.5 || s - psi(x_hat) ||_2^2 + beta/alpha ||s||_1
    % and we know the optimal solution is just soft thresholding of psi(x_hat)
    psi_xhat = psi(x_hat);
    
    % there are two ways to perform the soft-thresholding
    % s = max(0, abs(psi_xhat) - beta*alpha) .* sign(psi_xhat);
    s = max(0, psi_xhat - beta*alpha) - max(0, -psi_xhat - beta*alpha);
    
    % get x_hat from s
    x_hat = psi_adj(s);
    
    % print status
    obj = 0.5 * norm(vec(y-A(x_hat)))^2 + beta * norm(vec(psi(x_hat)),1);
    fprintf('iter = %d obj = %f\n', iter, obj);
    pause(0.05);
    
end
end

function y = vec(x)
y = x(:);
end

