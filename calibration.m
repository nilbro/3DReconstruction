function [ leftImageUndistorted,rightImageUndistorted,stereoParams ] = calibrate(image_left, image_right)

%% Putting in explicit values from YAML file

% Left Camera Matrix
IntrinsicMatrixLeft = transpose([ 1.03530811e+03            0       5.96955017e+02; 
    0           1.03508765e+03  5.20410034e+02; 
    0                   0           1 ]);

% Right Camera Matrix
IntrinsicMatrixRight = transpose([ 1.03517419e+03           0       6.88361877e+02; 
                                         0          1.03497900e+03  5.21070801e+02; 
                                         0                  0           1]);

% Rotation and Translation of camera
rotationOfCameraMatrix = [         1        1.94856493e-05  -1.52324792e-04; 
                            -1.95053162e-05       1         -1.29114138e-04; 
                             1.52322275e-04 1.29117107e-04          1. ];
translationOfCameraMatrix = [ -4.14339018e+00 -2.38197036e-02 -1.90685259e-03 ];

% radial distortion parameter left
radialDistortionLeft = [-5.95157442e-04 -5.46629308e-04];

% radial distortion parameter right
radialDistortionRight = [-2.34280655e-04 -7.68933969e-04];

% left camera parameters
cameraParamsLeft = cameraParameters('IntrinsicMatrix',IntrinsicMatrixLeft,'RadialDistortion',radialDistortionLeft);

% right camera parameters
cameraParamsRight = cameraParameters('IntrinsicMatrix',IntrinsicMatrixRight,'RadialDistortion',radialDistortionRight);

% Stereo Camera Parameters
stereoParams = stereoParameters(cameraParamsLeft,cameraParamsRight,rotationOfCameraMatrix,translationOfCameraMatrix);

% Undistort stereo pair
[leftImageUndistorted,rightImageUndistorted] = rectifyStereoImages(image_left,image_right,stereoParams,'OutputView','valid');

end


