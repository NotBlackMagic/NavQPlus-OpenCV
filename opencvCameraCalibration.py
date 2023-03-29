import cv2 as cv
import numpy as np
import glob

# https://learnopencv.com/camera-calibration-using-opencv/
# https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/
 
# Defining the dimensions of checkerboard
CHECKERBOARD = (7, 10)

# Termination criteria for refining the detected corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints_l = []
imgpoints_r = []
 
 # Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
 
# Extracting path of individual image stored in a given directory
path_l = "./data/stereoL/"
path_r = "./data/stereoR/"
for i in range(8):
	img_l = cv.imread(path_l + "img%d.png" % i)
	img_r = cv.imread(path_r + "img%d.png" % i)
	gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
	gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

	output_l = img_l.copy()
	output_r = img_r.copy()

	# Find the chess board corners
	# If desired number of corners are found in the image then ret = true
	ret_l, corners_l = cv.findChessboardCorners(output_l, CHECKERBOARD, None) # cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
	ret_r, corners_r = cv.findChessboardCorners(output_r, CHECKERBOARD, None)
		
	# If desired number of corner are detected, we refine the pixel coordinates and display them on the images of checker board
	if ret_l == True and ret_r == True:
		objpoints.append(objp)
		
		# refining pixel coordinates for given 2d points.
		cv.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
		cv.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
		
		# Draw and display the corners
		cv.drawChessboardCorners(output_l, CHECKERBOARD, corners_l, ret_l)
		cv.drawChessboardCorners(output_r, CHECKERBOARD, corners_r, ret_r)
		cv.imshow("Corners Left", output_l)
		cv.imshow("Corners Right", output_r)
		key = cv.waitKey(0)
		if key == 115:
			# Pressed s
			# Save to file
			print("Save captured images")
			cv.imwrite(path_l + "img%d_corners.png" % i, output_l)
			cv.imwrite(path_r + "img%d_corners.png" % i, output_r)

		imgpoints_l.append(corners_l)
		imgpoints_r.append(corners_r)

# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)
hL,wL = gray_l.shape[:2]
new_mtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))
 
# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)
hR,wR = gray_r.shape[:2]
new_mtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

# Step 2: Performing stereo calibration with fixed intrinsic parameters
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, new_mtxL, distL, new_mtxR, distR, gray_l.shape[::-1], criteria_stereo, flags)

# Step 3: Stereo Rectification
rectify_scale = 1
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv.stereoRectify(new_mtxL, distL, new_mtxR, distR, gray_l.shape[::-1], Rot, Trns, rectify_scale, (0, 0))

# Step 4: Compute the mapping required to obtain the undistorted rectified stereo image pair
Left_Stereo_Map = cv.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, gray_l.shape[::-1], cv.CV_16SC2)
Right_Stereo_Map = cv.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, gray_r.shape[::-1], cv.CV_16SC2)
 
print("Saving parameters ......")
cv_file = cv.FileStorage("improved_params2.xml", cv.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()
