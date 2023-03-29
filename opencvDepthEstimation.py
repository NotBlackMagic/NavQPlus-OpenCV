import numpy as np 
import cv2 as cv
import time
 
# https://nxp.gitbook.io/8mpnavq/dev-guide/software/opencv
# video/x-raw, format=YUY2, width=2592, height=1944, framerate=8/1
# video/x-raw, format=YUY2, width=1920, height=1080, framerate={ (fraction)15/1, (fraction)30/1 }
# video/x-raw, format=YUY2, width=1280, height=720, framerate={ (fraction)15/1, (fraction)30/1 }
# video/x-raw, format=YUY2, width=1024, height=768, framerate={ (fraction)15/1, (fraction)30/1 }
# video/x-raw, format=YUY2, width=720, height=576, framerate={ (fraction)15/1, (fraction)30/1 }
# video/x-raw, format=YUY2, width=720, height=480, framerate={ (fraction)15/1, (fraction)30/1 }
# video/x-raw, format=YUY2, width=640, height=480, framerate={ (fraction)15/1, (fraction)30/1 }
# video/x-raw, format=YUY2, width=320, height=240, framerate={ (fraction)15/1, (fraction)30/1 }
# video/x-raw, format=YUY2, width=176, height=144, framerate={ (fraction)15/1, (fraction)30/1 }
cap_l = cv.VideoCapture('v4l2src device=/dev/video3 ! video/x-raw, framerate={ (fraction)15/1, (fraction)30/1 }, width=320, height=240 ! appsink', cv.CAP_GSTREAMER)
cap_r = cv.VideoCapture('v4l2src device=/dev/video4 ! video/x-raw, framerate={ (fraction)15/1, (fraction)30/1 }, width=320, height=240 ! appsink', cv.CAP_GSTREAMER)

# Source: https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
# Reading the mapping values for stereo image rectification
# cv_file = cv.FileStorage("data/stereo_rectify_maps.xml", cv.FILE_STORAGE_READ)
# Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
# Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
# Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
# Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
# cv_file.release()

def nothing(x):
	pass

cv.namedWindow('disp', cv.WINDOW_NORMAL)
cv.resizeWindow('disp', 600, 600)

cv.createTrackbar('numDisparities','disp', 1, 17, nothing)
cv.createTrackbar('blockSize', 'disp', 5, 50, nothing)
cv.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
cv.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv.createTrackbar('minDisparity', 'disp', 5, 25, nothing)

# Creating an object of StereoBM algorithm
stereo = cv.StereoBM_create()

while True:
	# For frame timing
	start = time.time()
	
	# Capturing and storing left and right camera images
	ret_l, frame_l = cap_l.read()
	ret_r, frame_r = cap_r.read()

	# Proceed only if the frames have been captured
	if ret_l and ret_r:
		frame_r_gray = cv.cvtColor(frame_r,cv.COLOR_BGR2GRAY)
		frame_l_gray = cv.cvtColor(frame_l,cv.COLOR_BGR2GRAY)

		# Applying stereo image rectification on the left image
		# Left_nice= cv.remap(	frame_l_gray,
		#						Left_Stereo_Map_x,
		#						Left_Stereo_Map_y,
		#						cv.INTER_LANCZOS4,
		#						cv.BORDER_CONSTANT,
		#						0)

		# Applying stereo image rectification on the right image
		# Right_nice= cv.remap(	frame_r_gray,
		#						Right_Stereo_Map_x,
		#						Right_Stereo_Map_y,
		#						cv.INTER_LANCZOS4,
		#						cv.BORDER_CONSTANT,
		#						0)
		
		Left_nice = frame_l_gray
		Right_nice = frame_r_gray

		# Updating the parameters based on the trackbar positions
		numDisparities = cv.getTrackbarPos('numDisparities', 'disp') * 16
		blockSize = cv.getTrackbarPos('blockSize', 'disp') * 2 + 5
		preFilterType = cv.getTrackbarPos('preFilterType', 'disp')
		preFilterSize = cv.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
		preFilterCap = cv.getTrackbarPos('preFilterCap', 'disp')
		textureThreshold = cv.getTrackbarPos('textureThreshold', 'disp')
		uniquenessRatio = cv.getTrackbarPos('uniquenessRatio', 'disp')
		speckleRange = cv.getTrackbarPos('speckleRange', 'disp')
		speckleWindowSize = cv.getTrackbarPos('speckleWindowSize', 'disp') * 2
		disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff', 'disp')
		minDisparity = cv.getTrackbarPos('minDisparity', 'disp')

		# Setting the updated parameters before computing disparity map
		stereo.setNumDisparities(numDisparities)
		stereo.setBlockSize(blockSize)
		stereo.setPreFilterType(preFilterType)
		stereo.setPreFilterSize(preFilterSize)
		stereo.setPreFilterCap(preFilterCap)
		stereo.setTextureThreshold(textureThreshold)
		stereo.setUniquenessRatio(uniquenessRatio)
		stereo.setSpeckleRange(speckleRange)
		stereo.setSpeckleWindowSize(speckleWindowSize)
		stereo.setDisp12MaxDiff(disp12MaxDiff)
		stereo.setMinDisparity(minDisparity)

		# Calculating disparity using the StereoBM algorithm
		disparity = stereo.compute(Left_nice, Right_nice)
		# NOTE: Code returns a 16bit signed single channel image,
		# CV_16S containing a disparity map scaled by 16. Hence it 
		# is essential to convert it to CV_32F and scale it down 16 times.

		# Converting to float32 
		disparity = disparity.astype(np.float32)

		# Scaling down the disparity values and normalizing them 
		disparity = (disparity / 16.0 - minDisparity) / numDisparities

		# Displaying the disparity map
		cv.imshow("disp", disparity)
		cv.imshow("Left image", frame_l)
		cv.imshow("Right image", frame_r)
		
		# For frame timing
		end = time.time()
		delta = (end - start)
		rate = 1 / delta
		delta = delta * 1000
		print("Frame time: %d; Rate: %d" % (delta, rate))

		# Close window using esc key
		key = cv.waitKey(1)
		if key == 113:
			# Pressed q
			break

	else:
		print("Can't receive frame from camera (stream end?). Exiting ...")
		break

print("Saving depth estimation parameters ......")

cv_file = cv.FileStorage("../depth_estmation_params_py.xml", cv.FILE_STORAGE_WRITE)
cv_file.write("numDisparities", numDisparities)
cv_file.write("blockSize", blockSize)
cv_file.write("preFilterType", preFilterType)
cv_file.write("preFilterSize", preFilterSize)
cv_file.write("preFilterCap", preFilterCap)
cv_file.write("textureThreshold", textureThreshold)
cv_file.write("uniquenessRatio", uniquenessRatio)
cv_file.write("speckleRange", speckleRange)
cv_file.write("speckleWindowSize", speckleWindowSize)
cv_file.write("disp12MaxDiff", disp12MaxDiff)
cv_file.write("minDisparity", minDisparity)
cv_file.write("M", 39.075)
cv_file.release()
