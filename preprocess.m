function  preprocessedImage  = preprocess(image)

%% This function preprocesses the image before generating disparity map

% Homomorphic filtering to adjust the light intensity

image = im2double(image);
image = log(1 + image);
M = 2*size(image,1) + 1;
N = 2*size(image,2) + 1;
sigma = 10;

[X, Y] = meshgrid(1:N,1:M);
centerX = ceil(N/2); 
centerY = ceil(M/2); 
gaussianNumerator = (X - centerX).^2 + (Y - centerY).^2;
H = exp(-gaussianNumerator./(2*sigma.^2));
H = 1 - H; 
H = fftshift(H);
If = fft2(image, M, N);
Iout = real(ifft2(H.*If));
Iout = Iout(1:size(image,1),1:size(image,2));
imageHomomorphic = exp(Iout) - 1;

% Integral filtering to extract vertical and horizontal features

intImage = integralImage(imageHomomorphic);
horiH = integralKernel([1 1 4 3; 1 4 4 3],[-1, 1]);
vertH = horiH.';

horiResponse = integralFilter(intImage,horiH);
vertResponse = integralFilter(intImage,vertH);

% Fuse horizontal and vertical responses for feature enhancement
hor_ver = imfuse(horiResponse,vertResponse,'falsecolor','Scaling','none','ColorChannels','green-magenta');
preprocessedImage = rgb2gray(hor_ver);

end