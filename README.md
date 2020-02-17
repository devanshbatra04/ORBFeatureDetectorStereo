# StereoVision_for_uncalibrated_images
Image rectification and depth perception is straight forward in case of rectified images (when the camera parameters are known) but not in the case of uncalibrated images.

In order to obtain a fairly accurate disparity map, the stereo images must be rectified.

Rectification of Uncalibrated Images ->
1. Compute the features and there descriptors in the stereo images (ORB in the present code).
2. Match the features computed above.
3. Compute the Fundamental Matrix using the above computed feature matches.
4. Now using the fundamental matrix, calibrate the images using the function stereoRectifyUncalibrated.
