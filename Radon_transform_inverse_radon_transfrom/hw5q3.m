clear
close all
%% Load image
img = phantom('Modified Shepp-Logan',256);
M = 256;
N = linspace(10,180,18); % number of angles
siz = size(img);

%% Simulate different angles
err = zeros(18,1);

for i = 1:2%numel(N)
    %disp(i)
    Theta = 0:180/(N(1,i)-1):180;
    rad = radon(img, Theta);
    imgrec = least_square(rad, Theta, siz(1));
    if i == 1 || i == 9 || i == 18
        figure
        imshow(imgrec);
        str = sprintf('Reconstructed Image for N = %d',i*10);
        ylabel('x'),xlabel('y'),title(str);
    end
    err(i,1) = error(img, imgrec);
end

figure
plot(N,err);
title('Error vs. N')
ylabel('Error')
xlabel('N')
%%
function x = least_square(rad, theta, s)
    ra_siz = size(rad);
    ira = iradon(rad, theta, 'linear', 'none', s);
    
    %%constants
    cgs_iters = 100;                 % Number of conjugate gradient descent iterations
    cgs_tol = 1e-4;                 % CGS tolerance.
    lambda_reg = 1;             % Regularization parameter for regularized least squares.

    %%function handle
    Aradon = @(img) radon(reshape(img, s, s), theta);
    Aradon_adjoint = @(rad_img) iradon(reshape(rad_img, ra_siz), theta, 'linear', 'none', 1, s);


    Dxadj = @(z) vec(Dxadjoint(z));
    Dxfor = @(z) vec(Dx(z));
    Dyadj = @(z) vec(Dyadjoint(z));
    Dyfor = @(z) vec(Dy(z));
    
%     A = @(x) vec(Aradon_adjoint(Aradon(x))) ;
    A = @(x) vec(Aradon_adjoint(Aradon(x))) + lambda_reg * ((Dxadj(Dxfor(x)))+(Dyadj(Dyfor(x))));
    b = vec(ira);
    
    x = cgs(A, b, cgs_tol, cgs_iters);
    x = reshape(x, s, s);
    
end

function e = error(img, imgrec)
    e = -20*log10( norm(img(:) - imgrec(:))/norm(img(:)));
end

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
