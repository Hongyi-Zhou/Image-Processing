function [x] = OMP(y, A, K, tol)
%Implements OMP to solve: min ||y - Ax||^2 st. ||x||_0 \le K
% Input
% y   - vector of size Mx1
% A   - matrix of size MxN
% K   - sparsity of solution
% tol - residual tolerance
% Output
% x - K-sparse vector of size Nx1
%
    x = zeros(size(A,2),1);

end