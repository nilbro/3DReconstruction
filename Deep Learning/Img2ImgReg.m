clc;
clear;
%% Image to Image Regression using UNet. MPI-Sintel Dataset has been used for Training

% Disparity Images Datastore (Train)
dispDir = '/home/bro/Downloads/MPI-Sintel-stereo-training-20150305/training/disparities/market_temple';
dispImages = imageDatastore(dispDir,'FileExtensions','.png','IncludeSubfolders',true);

% Input Images Datastore (Train)
imagesDir = '/home/bro/Downloads/MPI-Sintel-stereo-training-20150305/training/clean_left/market_temple';
trainImagesDir = fullfile(imagesDir);
trainImages = imageDatastore(trainImagesDir,'FileExtensions','.png','IncludeSubfolders',true);

% Disparity Images Datastore (Test)
dispDirTest = '/home/bro/Downloads/MPI-Sintel-stereo-training-20150305/training/disparities/cave_2';
dispImagesTest = imageDatastore(dispDir,'FileExtensions','.png','IncludeSubfolders',true);

% Input Images Datastore (Test)
imagesDirTest = '/home/bro/Downloads/MPI-Sintel-stereo-training-20150305/training/clean_left/cave_2';
trainImagesDir = fullfile(imagesDir);
trainImagesTest = imageDatastore(trainImagesDir,'FileExtensions','.png',IncludeSubfolders',true);

% Display images from Datastore
im_orig = trainImages.readimage(1);
im_disp = dispImages.readimage(1);

imshow(im_orig);
title('Clean Image - Final Result');
figure; imshow(im_disp);
title('Blurred Image - Input');

%% Random Patching to increase Training Data

% Preprocessing options for Data Augmentation (Rotation, Reflection)
augmenter = imageDataAugmenter( ...
    'RandRotation',@()randi([0,1],1)*90, ...
    'RandXReflection',true);
patchSize = [40 40];

% Generate Patch Datastore
patchds = randomPatchExtractionDatastore(dispImages,trainImages,patchSize, ....
'PatchesPerImage',64, ...
'DataAugmentation',augmenter);

miniBatchSize = 64;
patchds.MiniBatchSize = miniBatchSize;

%% Build Network

% Segmentation Layer UNet of depth 3 has been considered
 lgraph = unetLayers([40 40 3] , 3,'encoderDepth',3);
 
% Convert Segmentation Network into a Classification Network
 lgraph = lgraph.removeLayers('Softmax-Layer');
 lgraph = lgraph.removeLayers('Segmentation-Layer');
 lgraph = lgraph.addLayers(regressionLayer('name','regressionLayer'));
 lgraph = lgraph.connectLayers('Final-ConvolutionLayer','regressionLayer');


%% Train Network 

% Training Configuration
maxEpochs = 40;
epochIntervals = 1;
initLearningRate = 0.01;
learningRateFactor = 0.01;
l2reg = 0.0001;
options = trainingOptions('sgdm', ...
    'Momentum',0.9, ...
    'InitialLearnRate',initLearningRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',10, ...
    'LearnRateDropFactor',learningRateFactor, ...
    'L2Regularization',l2reg, ...
    'ValidationData',{trainImagesTest,dispImagesTest}, ...
    'MaxEpochs',maxEpochs ,...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThresholdMethod','l2norm', ...
    'Plots','training-progress', ...
    'GradientThreshold',0.01);

% Train and Save Network
 modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
 net = trainNetwork(patchds,lgraph,options);
 save(['trainedNet-' modelDateTime '-Epoch-' num2str(maxEpochs*epochIntervals) ...
            'ScaleFactors-' num2str(234) '.mat'],'net','options');
        

%% Test Network

% Read Test Image
testImage = imread('..');

% Check Output
 testOutput = activations(net,testImage,'regressionoutput');
 figure; imshow(testOutput)
 Iapprox = rescale(testOutput);
 Iapprox = im2uint8(Iapprox);
 imshow(Iapprox)
title('Disparity Image')

