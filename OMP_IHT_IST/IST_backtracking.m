function [x_hat] = IST_backtracking(y, beta, psi, psi_adj, A, A_adj, max_iter)
%ITERATIVE_SOFT_THRESHOLDING 
% return x_hat = argmin_x  0.5 ||y - A(x)||^2_2 + beta || phi(x) ||_1 

x_hat = zeros(size(A_adj(y)));

%fig = figure;

% demo the use of anaoymous function
gradient = @(x) -A_adj(y-A(x));
least_square = @(x) 0.5*sum(vec(y-A(x)).^2);

t0 = 1;
t_beta = 0.9;

for iter = 1:max_iter
    % this is basically a fixed point method (or proximal gradient descent)
    % first we perform the typical gradient descent for the first part
    
    % use backtracking line search to find the step size
    % for details, see the page 6 and 11 of http://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-grad.pdf
    t = t0;
    g = gradient(x_hat);
    while true
        Gt = (x_hat - psi_adj(soft_thresholding(psi(x_hat-t*g), t*beta)))/t;
        test_lhs = least_square(x_hat - t*Gt);
        test_rhs = least_square(x_hat) - t*vec(g)'*vec(Gt) + 0.5*t*sum(vec(Gt).^2);
        if test_lhs > test_rhs
            t = t_beta * t;
        else
            break;
        end
    end
    % update x_hat 
    x_hat = x_hat - t * Gt; 
    
    % print status
    obj = 0.5 * norm(vec(y-A(x_hat)))^2 + beta * norm(vec(psi(x_hat)),1);
    fprintf('iter = %d obj = %f\n', iter, obj);
    pause(0.05);
end
end

function y = vec(x)
y = x(:);
end

function y = soft_thresholding(x, beta)
y = max(0, x - beta) - max(0, -x-beta);
end
