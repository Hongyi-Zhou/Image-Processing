clear;
close all;
%% Load image
img = phantom('Modified Shepp-Logan',256);
img = img/max(max(img));
theta = 0:1:180;
rad = radon(img, theta);
npad = 200;
ramlak_thres = 180;

%% filter before back projection

R_fbp = zeros(size(rad));
for idx = 1:length(theta)
	% Get a slice's Fourier transform
    R_slice = rad(:, idx);
    R_slice = [R_slice; zeros(npad, 1) ]';
    F_slice = fftshift(fft(R_slice));

	% Create ramp in Fourier domain
    ramp = -(length(F_slice)-1)/2:(length(F_slice)-1)/2;
    
    % Filter using ramlak filter
    ramp(1:ramlak_thres) = ramp(ramlak_thres);
    ramp(end-ramlak_thres+1:end) = ramp(end-ramlak_thres+1);
    F_slice = F_slice.*abs(ramp);

	% Take inverse FFT, but remember to use ifftshift instead of fftshift
    R_fbp_slice = real(ifft(ifftshift(F_slice)));

	% Now save the Radon slice
    R_fbp(:, idx) = R_fbp_slice(1:size(rad, 1))';
end

%% back projection
fbp_img = zeros(size(rad,1));
temp = zeros(size(rad,1));
for i = 0:1:180
    %smearing along y axis
    temp = ((R_fbp(:,i+1))*ones(1, size(rad, 1)))';
    temp = imrotate(temp, i, 'bilinear', 'crop');
    fbp_img = fbp_img + temp;
end

fbp_img = fbp_img(56:311, 56:311);
fbp_img = fbp_img/max(max(fbp_img));
% 
figure
imshow(img);
ylabel('x'),xlabel('y'),title('Original Image');

figure
imshow(fbp_img, []);
ylabel('x'),xlabel('y'),title('Filtered Back Projection Image');

figure
l = imabsdiff(img, fbp_img);
imshow(l, []);
ylabel('x'),xlabel('y'),title('Absolute Difference');