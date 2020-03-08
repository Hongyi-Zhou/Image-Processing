clear
close all
%% Load data
data = load("Q1.mat");
imblur = data.imblur;
imsharp = data.imsharp;
cgs_iters = 350;                 % Number of conjugate gradient descent iterations
cgs_tol = 1e-6;                 % CGS tolerance.
lambda_reg = 5e-01;

%% Function Handle
A_forward = @(z) vec(conv2(imsharp, reshape(z,15,15), 'valid'));

A_adjoint = @(z) vec(conv2(imsharp(end:-1:1, end:-1:1), reshape(z,242,242), 'valid'));

%% cgs
A1 = @(z) A_adjoint(A_forward(z)) + lambda_reg * z;
b = A_adjoint(imblur(:));

kernel = cgs(A1, b, cgs_tol, cgs_iters);
kernel = reshape(kernel, 15, 15);
imagesc(kernel)
