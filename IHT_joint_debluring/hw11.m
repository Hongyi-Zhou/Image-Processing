clear
close all
%% Load data
data = load("hw11.mat");
y2 = data.y2;
y1 = data.y1;
k1 = data.k1;
k2 = data.k2;
[Hb, Wb, Cb] = size(y1);
K = int32(0.2*Hb*Wb);

%% deblurring
%jointly
A_forward = @(z) forward(z, k1, k2);
A_adjoint = @(z) adjoint(z, k1, k2);
y = cat(1,y1,y2);

%separately
% A_forward = @(z) forward_sep(z, k1, k2);
% A_adjoint = @(z) adjoint_sep(z, k1, k2);
% y = y2;

x_init = A_adjoint(y);

%k = [0 -1 0; -1 5 -1; 0 -1 0];
% for ch = 1:3
%     x_init(:,:,ch) = conv2(x_init(:,:,ch), k, 'same');
% end

%% function handle for phi
%impainting initializa x_init with filling values
%debluring initialize x_init with adjoint
% 
[~, cbook] = wavedec2(x_init, 8, 'db4');
[~, cbook1] = wavedec2(x_init(:,:,1), 8, 'db4');

Psi = @(x) wavedec2(x, 8, 'db4');
Psi_adjoint = @(x) waverec2(x, cbook, 'db4');

%% solving
x = IHT_deblur(y, K, Psi, Psi_adjoint, A_forward, A_adjoint, 200, x_init);
%x = IST(y, 0.01, Psi, Psi_adjoint, A_forward, A_adjoint, 200, x_init);

%% functions
function y = forward(x, k1, k2)
    for ch=1:3 
        y1(:, :, ch) = conv2(x(:, :, ch), k1, "valid"); 
        y2(:, :, ch) = conv2(x(:, :, ch), k2, "valid"); 
    end
    y = cat(1,y1,y2);
end

function x = adjoint(y, k1, k2)
    l = floor(size(y,1)/2); 
    y1 = y(1:l,:,:);
    y2 = y(l+1:end,:,:);
    for ch=1:3 
        x1(:, :, ch) = conv2(y1(:, :, ch), k1(end:-1:1, end:-1:1), "full"); 
        x2(:, :, ch) = conv2(y2(:, :, ch), k2(end:-1:1, end:-1:1), "full"); 
    end
    x = x1 + x2;
end

function y = forward_sep(x, k1, k2)
    for ch=1:3 
        y(:, :, ch) = conv2(x(:, :, ch), k1, "valid"); 
    end
end

function x = adjoint_sep(y, k1, k2)
    for ch=1:3 
        x(:, :, ch) = conv2(y(:, :, ch), k1(end:-1:1, end:-1:1), "full"); 
    end
end

%% IHT
function xstar = IHT_deblur(y, K, Psi, PsiAdj, A, AAdj, MaxIter, x_init)
    iter = 0;
    eta = 0.5;
    [s, cbook] = wavedec2(x_init, 8, 'db4');
    
    while (iter<MaxIter)
        gradient = Psi(AAdj(y - A(PsiAdj(s))));
        s = s + eta * gradient;
        iter = iter + 1;
        
        %hard thresholding
        s0 = sort(abs(s), 'descend');
        shat = s0(K);
        s(s<shat & s>-shat) = 0;
        if (mod(iter,10)==0)
            disp(iter)
        end
    end
    xstar = waverec2(s, cbook, 'db4');
end
