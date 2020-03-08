clear
close all
%% Load data
data = load("hw9.mat");
D = data.D;
blurred = data.blurred;
corrupted = data.corrupted;
k0 = data.k0;
mask = data.mask;
mosaik = data.mosaik;
super = data.super;
lambda_reg = 0;
[Hb, Wb] = size(blurred);
K = int32(0.1*Hb*Wb);
%% gradient function handles
Dxadj = @(z) Dx_adjoint(z);
Dxfor = @(z) Dx(z);
Dyadj = @(z) Dy_adjoint(z);
Dyfor = @(z) Dy(z);

%% deblurring

% yy = blurred;
% A_forward = @(z) conv2(reshape(z, Hb, Wb), k0, 'same');
% A_adjoint = @(z) conv2(reshape(z, Hb, Wb), k0(end:-1:1, end:-1:1), 'same');
% x_init = A_adjoint(yy);
% k = [0 -1 0; -1 5 -1; 0 -1 0];
% for i = 1:3
%     x_init = conv2(x_init, k, 'same');
% end

%% inpainting

% yy = corrupted;
% 
% A_forward = @(x) x.*mask;
% A_adjoint = @(x) x.*mask;
% k = ones(6)/36;
% x_init = conv2(yy, k, 'same');

% [x y] = meshgrid(1:size(yy,2),1:size(yy,1));
% [xi yi] = meshgrid(1:0.5:size(yy,2),1:0.5:size(yy,1));
% x_ = griddata(x,y,double(yy),xi,yi,'cubic');
% 
% F = griddedInterpolant(double(x_));
% xq = (0:2:size(x_,1))';
% yq = (0:2:size(x_,2))';
% x_init = double(F({xq,yq}));


%% demosaicking

yy = mosaik;
yy = im2double(yy);
A_forward = @(z) AM(z);
A_adjoint = @(z) AM_adjoint(z);
k = ones(4)/16;
x_init = A_adjoint(yy);
x_init(:,:,1) = conv2(x_init(:,:,1), k, 'same')+0.1;
x_init(:,:,2) = conv2(x_init(:,:,2), k, 'same')+0.1;
x_init(:,:,3) = conv2(x_init(:,:,3), k, 'same')+0.1;

%% function handle for phi
%impainting initializa x_init with filling values
%debluring initialize x_init with adjoint

[~, cbook] = wavedec2(x_init, 8, 'db4');
[~, cbook1] = wavedec2(x_init(:,:,1), 8, 'db4');

Psi = @(x) wavedec2(x, 8, 'db4');
Psi_adjoint = @(x) waverec2(x, cbook, 'db4');
%% solving
%x = IHT_deblur(yy, K, Psi, Psi_adjoint, A_forward, A_adjoint, Dxadj, Dxfor, Dyadj, Dyfor, 100, x_init, lambda_reg);
%x = IHT_inpaint(yy, K, Psi, Psi_adjoint, A_forward, A_adjoint, 200, x_init);
x = IHT_mosaik(yy, K, Psi, Psi_adjoint, A_forward, A_adjoint, 400, x_init);

%% Demosaiking
function y = AM(I)
    y = zeros(size(I,1),size(I,2));
    for i = 1:size(I,1)
        for j = 1:size(I,2)

            if(mod(i,2)==1 && mod(j,2)==1)
                y(i,j) = I(i,j,3);
            elseif (mod(i,2)==0 && mod(j,2)==0)
                y(i,j) = I(i,j,1);
            else
                y(i,j) = I(i,j,2);
            end
            
        end
    end
end

function I = AM_adjoint(y)
    I = zeros(size(y,1),size(y,2),3);
    for i = 1:size(y,1)
        for j = 1:size(y,2)
            
            if(mod(i,2)==1 && mod(j,2)==1)
                I(i,j,3) = y(i,j);
            elseif (mod(i,2)==0 && mod(j,2)==0)
                I(i,j,1) = y(i,j);
            else
                I(i,j,2) = y(i,j);
            end
            
        end
    end
end

%% Dx
function y = Dx(I)
    y = I(:, 2:end)-I(:, 1:end-1);
end

%% Dx adjoint
function I = Dx_adjoint(y)
    I = zeros(size(y)+[0 1]);
    I(:, 2:end) = y;
    I(:, 1:end-1) = I(:, 1:end-1) - y;
end

%% Dy
function y = Dy(I)
    y = I(2:end, :)-I(1:end-1, :);
end

%% Dy adjoint
function I = Dy_adjoint(y)
    I = zeros(size(y)+[1 0]);
    I(2:end,:) = y;
    I(1:end-1,:) = I(1:end-1,:) - y;
end

%% gradient IHT
function xstar = IHT_deblur(y, K, Psi, PsiAdj, A, AAdj, Dxadj, Dxfor, Dyadj, Dyfor, MaxIter, x_init, lambda_reg)
    iter = 0;
    eta = 0.5;
    [s, cbook] = wavedec2(x_init, 8, 'db4');
    
    while (iter<MaxIter)
        gradient = Psi(AAdj(y - A(PsiAdj(s)))) + lambda_reg*(Psi(Dxadj(Dxfor(PsiAdj(s)))) + Psi(Dyadj(Dyfor(PsiAdj(s)))));
        s = s + eta * gradient;
        iter = iter + 1;
        
        %hard thresholding
        s0 = sort(abs(s), 'descend');
        shat = s0(K);
        s(s<shat & s>-shat) = 0;
        if (mod(iter,50)==0)
            disp(iter)
        end
    end
    xstar = waverec2(s, cbook, 'db4');
end

%% IHT
function xstar = IHT_inpaint(y, K, Psi, PsiAdj, A, AAdj, MaxIter, x_init)
    iter = 0;
    eta = 1;
    [s, cbook] = wavedec2(x_init, 8, 'db4'); %find initial s
    
    while (iter<MaxIter)
        gradient = Psi(AAdj(y - A(PsiAdj(s)))); %calculate gradient
        s = s + eta * gradient; %update
        iter = iter + 1; 
        
        %hard thresholding
        s0 = sort(abs(s), 'descend');
        shat = s(K);
        s(s<shat & s>-shat) = 0; %keep top k wavelet coefficients
        if (mod(iter,50)==0)
            disp(iter)
        end
    end
    xstar = waverec2(s, cbook, 'db4');
end

%% Three Chanel IHT
function xstar = IHT_mosaik(y, K, Psi, PsiAdj, A, AAdj, MaxIter, x_init)
    iter = 0;
    eta = 1;
    [s, cbook] = wavedec2(x_init, 8, 'db4');
%     [s1, cbook] = wavedec2(x_init(:,:,1), 8, 'db4');
%     [s2, cbook] = wavedec2(x_init(:,:,2), 8, 'db4');
%     [s3, cbook] = wavedec2(x_init(:,:,3), 8, 'db4');
    
    while (iter<MaxIter)
        gradient = Psi(AAdj(y - A(PsiAdj(s))));
        s = s + eta * gradient;
        iter = iter + 1;
        
        %hard thresholding
        r = PsiAdj(s);
        [s1, cbook1] = wavedec2(r(:,:,1), 8, 'db4');
        [s2, cbook1] = wavedec2(r(:,:,2), 8, 'db4');
        [s3, cbook1] = wavedec2(r(:,:,3), 8, 'db4');
        s = s1+s2+s3;
        s0 = sort(abs(s), 'descend');
        shat = s0(K);
        idx = find(s<shat & s>-shat);
        %disp(size(idx));
        s1(idx) = 0;
        s2(idx) = 0;
        s3(idx) = 0;

        rec1 = waverec2(s1, cbook1, 'db4');
        rec2 = waverec2(s2, cbook1, 'db4');
        rec3 = waverec2(s3, cbook1, 'db4');
        rec = zeros(size(rec1,1),size(rec1,2),3);
        %disp(size(rec1));
        %disp(size(rec));
        rec(:,:,1) = rec1;
        rec(:,:,2) = rec2;
        rec(:,:,3) = rec3;
        s = Psi(rec);
        
        if (mod(iter,50)==0)
            disp(iter)
        end
    end
    xstar = waverec2(s, cbook, 'db4');
end