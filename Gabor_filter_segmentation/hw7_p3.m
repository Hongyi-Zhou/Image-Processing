clear
close all
%% Load image
img = imread('one.tiff'); 
img = double(img)/255;

%% constants
thresh1 = 0.25;
thresh2 = 0.19;%0.25;

%% gabor
%1 32, 0, 1/12
%2 2, pi/4, 30
g1 = gabor(32, 0, 1/12);
g2 = gabor(2, pi/4, 30);

imgconv1 = conv2(img, g1, 'same');
imgconv2 = conv2(img, g2, 'same');

img1 = abs(imgconv1);
img2 = abs(imgconv2);

img1 = img1/max(max(img1));
img2 = img2/max(max(img2));

ret = zeros(size(img1));
im = conv2(img2, ones(40)/1600, 'same');
im = im/max(max(im));

for i = 1:size(img1,1)
    for j = 1:size(img1,1)
        if im(i,j) < thresh2 
            if img1(i,j)> thresh1
                ret(i,j) = 1;
            end
        end
        
    end
end


imshow(ret); title('Segmented');

%% Gabor filter
function f = gabor(sigma, theta, v0)
    [x,y] = meshgrid(-75:75);
    
    f = exp(-(x.^2+y.^2)/(2*sigma^2)).*exp(-2j*pi*v0*(x*cos(theta)+y*sin(theta)));
end
