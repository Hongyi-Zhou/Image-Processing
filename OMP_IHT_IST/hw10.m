clear
close all
%% Load data
data = load("hw10.mat");
y = data.y;
K = data.K;
A = data.A;

[xstar_A, omega] = OMP(y,A,K);

%% Implements OMP to solve: min ||y - Ax||^2 st. ||x||_0 \le K
% Input
% y - vector of size Mx1
% A - matrix of size MxN
% K - sparsity of solution
% Output
% x - K-sparse vector of size Nx1

function [xx, omega] = OMP(y, A, K)
    [~, n] = size(A);
    r = y; %residual
    xx = zeros(n,1);
    omega = zeros(K, 1);
    A_omega = [];
    
    for i = 1:K
        [~, ind] = max(abs(A'*r));
        omega(i) = ind;
        A_omega = [A_omega A(:,ind)];
        x = A_omega \ y; 
        r = y - A_omega * x; 
    end
    
    for i = 1:K
        xx(omega(i)) = x(i); %x_sparse(i).value;
    end

end

