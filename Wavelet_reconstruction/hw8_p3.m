clear
close all
%% Load image
targetSize = [2048 2048];
Kfrac = 0.01;
ss = [32, 64, 128, 256, 512, 1024, 2048];
error = zeros(16,length(ss));

for i = 1:16
    img = imread(sprintf('%01d.jpg', i)); 
    img = rgb2gray(img);
    img = double(img)/255;
    %[x,y,w,h] = center(size(img),targetSize);
    I = imcrop(img,[1,1,targetSize(1)-1,targetSize(2)-1]);
    %processed(1,i) = I;
    %subplot(4,4,i);
    %imshow(I);

    for j = 1:length(ss)
        N0 = ss(j);
        %downsample
        I1 = imresize(I,[N0 N0]);
%         figure;
%         imshow(I1);
    
        dwtmode("per");
        [s0, cbook] = wavedec2(I1, log2(N0), 'db4');
        
        s = abs(sort(s0, 'descend'));
        K = int32(Kfrac*N0*N0);
        shat = s(K);

        s0(s0<shat & s0>-shat) = 0;
        imghat = waverec2(s0, cbook, 'db4');

        error(i,j) = norm(I1-imghat,2)/norm(I1,2);
        disp(i)
        disp(j)
    end
end

err = sum(error);
figure;
plot(ss,err);
xlabel('N0')
ylabel('average normalized reconstruction error/ SNR')

function [x, y, w, h] = center(a,b)
    x = int16(a(1)/2 - b(1)/2);
    y = int16(a(2)/2 - b(2)/2);
    w = int16(b(1) - 1);
    h = int16(b(2) - 1);
end