clear
close all
%% Load image
load('hw3_problem3.mat')
img = imread('cameraman.tif'); 
img = double(img)/255;

%% global constants
cgs_iters = 350;                 % Number of conjugate gradient descent iterations
cgs_tol = 1e-6;                 % CGS tolerance.

lambda_reg1 = 1e-02;             % Regularization parameter for regularized least squares.
lambda_reg2 = 1e-04;             % Regularization parameter for regularized least squares.

Omega = omega1;
% generate forward and adjoint function handles
Aforward = @(z) AOmega(z, Omega);
Aadj = @(z) AOmega_adjoint(z, Omega);

% Create gradient operators too
Dxadj = @(z) vec(Dxadjoint(z));
Dxfor = @(z) vec(Dx(z));
Dyadj = @(z) vec(Dyadjoint(z));
Dyfor = @(z) vec(Dy(z));


% #4
% now we wish to minimize 0.5*\|y - Ax\|^2 + lamda*|x|^2. 
% We solve this by setting gradient to zero, which gives us,
%
%       Aadj(Aforward(x))+lamda*x2 = Aadj(y);
%


% #5
% Now we wish to solve 0.5*\|y - Ax\|^2 + lambda_reg*(\|Dx.x\|^2 +
% \|Dy.x\|^2).
% We solve this by setting gradient to zero, which gives us,
%
% Aadj(Aforward(x)) + lambda_reg*(Dxadjoint(Dxforward(x))) +  lambda_reg*(Dyadjoint(Dyforward(x)))= Aadj(y);
% This is of the form A(x) = b, where A = ((Aadj(Aforward(.)) + lambda_reg*I) and 
% b = Adj(y). This can be now solved using CGS. CGS can also take a
% function operator as input, and hence we can create a new function
% handle.

% Create gradient operators too

A1 = @(z) Aadj(vec(Aforward(z))) + lambda_reg1 * z;
A2 = @(z) Aadj(vec(Aforward(z))) + lambda_reg2 * ((Dxadj(Dxfor(z)))+(Dyadj(Dyfor(z))));

b = Aadj(img(:));

% Now try recovering the image using cgs with norm on image
im_rec1 = cgs(A1, b, cgs_tol, cgs_iters);
im_rec1 = reshape(im_rec1, 256, 256);

% Now try recovering the image using cgs with norm on image gradients.
im_rec2 = cgs(A2, b, cgs_tol, cgs_iters);
im_rec2 = reshape(im_rec2, 256, 256);
% 
noisy_image = Aforward(vec(img));
noisy_image = reshape(noisy_image, 256, 256);
% 
% check_image = Dx(vec(img));
% check_image = Dxadj(check_image);
% check_image = reshape(check_image, 256, 256);
% 
% subplot(2, 2, 1); imshow(img); title('Original image');
subplot(1, 2, 1); imshow(noisy_image); title('Noisy image');
subplot(1, 2, 2); imshow(im_rec2); title('Recovered Photo2');
% subplot(2, 2, 4); imshow(im_rec2); title('Recovered Photo2');
% figure(1);
% imshow(im_rec1);
% title('Recovered Photo1');
% 
% figure(2);
% imshow(im_rec2);
% title('Recovered Photo2');


%% Dx
function y = Dx(I)
    i = reshape(I, [256,256]);
    y = i(:, 2:end)-i(:, 1:end-1);
    y = vec(y);
end

%% Dx adjoint
function I = Dxadjoint(y)
    Y = reshape(y, [256,255]);
    I = zeros(256,256);
    I(:,1) = -Y(:,1);
    I(:,2:end-1) = Y(:,2:end) - Y(:,1:end-1);
    I(:,end) = Y(:,end);
    I = vec(I);
end

%% Dy
function y = Dy(I)
    i = reshape(I, [256,256]);
    y = i(2:end, :)-i(1:end-1, :);
    y = vec(y);
end

%% Dy adjoint
function I = Dyadjoint(y)
    Y = reshape(y, [255,256]);
    I = zeros(256,256);
    I(1,:) = -Y(1,:);
    I(2:end-1,:) = Y(2:end,:) - Y(1:end-1,:);
    I(end,:) = Y(end,:);
    I = vec(I);
end

%% problem3#2
function y = AOmega(x, omega)
    X = reshape(x,[256,256]);
    y = X;
    xind = sub2ind(size(X), omega(:,1), omega(:,2));
    y(xind) = 0;
    y = vec(y);
end

%% problem3#3
function x = AOmega_adjoint(y, omega)
    Y = reshape(y,[256,256]);
    x = Y;
    xind = sub2ind(size(Y), omega(:,1), omega(:,2));
    x(xind) = 0;
    x = vec(x);
end