
clc;
clear;
%% Read and undistort images

% Read distoted images
rightImage = imread('Right_Image.png');
leftImage = imread('Left_Image.png');

% Read GroundTruth images
rightImageGnd = imread('right_depth_map.tiff');
leftImageGnd = imread('left_depth_map.tiff');

% Calibrate cameras and undistort
[leftImageUndistorted,rightImageUndistorted,stereoParams] = calibration(leftImage,rightImage);
[leftImageUndistortedGnd,rightImageUndistortedGnd,stereoParams] = calibration(leftImageGnd,rightImageGnd);

%% Preprocess Images

% Proprocess input images
preprocessedLeftImage = preprocess(leftImageUndistorted);
preprocessedRightImage = preprocess(rightImageUndistorted);

% Grayscale Images
rightImageBWGnd = rgb2gray(rightImageUndistortedGnd);
leftImageBWGnd = rgb2gray(leftImageUndistortedGnd);

rightImageBW = rgb2gray(preprocessedRightImage);
leftImageBW = rgb2gray(preprocessedLeftImage);

%% Generate Disparity Map

% Disparity Range
disparityRange = [48 88];

% Generate Disparity Map by Semi-Global Matching
disparityMapGnd = disparitySGM(leftImageBWGnd, rightImageBWGnd,'DisparityRange', disparityRange,'UniquenessThreshold',0);
disparityMap = disparitySGM(preprocessedLeftImage, preprocessedRightImage,'DisparityRange', disparityRange,'UniquenessThreshold',0);

% Pad Disparity Map for reconstruction and comparison
disparityMap = padarray(disparityMap,[3 3],'replicate','post');
 
imshow(disparityMap,disparityRange)
 title('Disparity Map');
 colormap jet;
 colorbar; 

%% Reconstruct Scene

  xyzPoints = reconstructScene(disparityMap,stereoParams);
  points3D = xyzPoints ./ 100;
  ptCloud = pointCloud(points3D,'color',leftImageUndistorted);
  ptCloudOut = removeInvalidPoints(ptCloud);
  ptCloudOut = pcdenoise(ptCloudOut);
  pcshow(ptCloudOut)

%% Depth Estimation  

% Find connected regions(blobs) in the Integral Image
 blob = vision.BlobAnalysis('BoundingBoxOutputPort', true,'MinimumBlobAreaSource', 'Property','MinimumBlobArea', 50);
 [area,centroid,bboxes] = step(blob,imbinarize(leftImageBW));
 
 
% Find the centroids of detected blobs.
 centroids = [round(bboxes(:, 1) + bboxes(:, 3) / 2), ...
     round(bboxes(:, 2) + bboxes(:, 4) / 2)];
 
% Find the 3-D world coordinates of the centroids.
centroidsIdx = sub2ind(size(disparityMap), centroids(:, 2), centroids(:, 1));
 X = points3D(:, :, 1);
 Y = points3D(:, :, 2);
 Z = points3D(:, :, 3);
 centroids3D = [X(centroidsIdx)'; Y(centroidsIdx)'; Z(centroidsIdx)'];
 
% Find the distances from the camera in meters.
 dists = sqrt(sum(centroids3D .^ 2));
     
% Display the detected blobs and their distances.
 labels = cell(1, numel(dists));
 for i = 1:numel(dists)
     labels{i} = sprintf('%0.2f meters', dists(i));
 end
 figure;
 imshow(insertObjectAnnotation(leftImageUndistorted, 'rectangle', bboxes, labels));
 title('Distance From Camera');