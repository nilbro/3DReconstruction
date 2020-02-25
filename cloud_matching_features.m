
clc;
clear;
%% Read and undistort images
% Read Images

leftImage = imread('Left_Image.png');
rightImage = imread('Right_Image.png');

% Convert to Grayscale
leftImageBW = rgb2gray(leftImage);
rightImageBW = rgb2gray(rightImage);

% Calibrate and Undistort images
[leftImageUndistorted,rightImageUndistorted,stereoParams] = calibration(leftImageBW,rightImageBW);

% Preprocess Images
preprocessedLeftImage = preprocess(leftImageUndistorted);
preprocessedRightImage = preprocess(rightImageUndistorted);

%% Feature Matching
pointTracker = vision.PointTracker;

% KAZE features
pointsL = detectKAZEFeatures(preprocessedLeftImage);
pointsR = detectKAZEFeatures(preprocessedRightImage);

% Determine valid feature points
[featuresL,valid_pointsL] = extractFeatures(preprocessedLeftImage,pointsL);
[featuresR,valid_pointsR] = extractFeatures(preprocessedRightImage,pointsR);

% Match features
indexPairs = matchFeatures(featuresL,featuresR);

% Matched features in left and right images
matchedPointsL = valid_pointsL(indexPairs(:,1),:);
matchedPointsR = valid_pointsR(indexPairs(:,2),:);

% Sparse Reconstruction
worldPoints = triangulate(matchedPointsL,matchedPointsR,stereoParams);
pcshow(worldPoints)