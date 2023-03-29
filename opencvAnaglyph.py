import numpy as np
import cv2 as cv

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
cap_l = cv.VideoCapture('v4l2src device=/dev/video3 ! video/x-raw, framerate={ (fraction)15/1, (fraction)30/1 }, width=640, height=480 ! appsink', cv.CAP_GSTREAMER)
cap_r = cv.VideoCapture('v4l2src device=/dev/video4 ! video/x-raw, framerate={ (fraction)15/1, (fraction)30/1 }, width=640, height=480 ! appsink', cv.CAP_GSTREAMER)

# Define the codec and create VideoWriter object
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('C:/Users/fabia/Desktop/test.avi', fourcc, 20.0, (640,  480))

# Load calibration values
cv_file = cv.FileStorage("improved_params2.xml", cv.FILE_STORAGE_READ)
Left_Stereo_Map = [	cv_file.getNode("Left_Stereo_Map_x").mat(), cv_file.getNode("Left_Stereo_Map_y").mat()]
Right_Stereo_Map = [cv_file.getNode("Right_Stereo_Map_x").mat(), cv_file.getNode("Right_Stereo_Map_y").mat()]

# cap = cv.VideoCapture(0)
if not cap_l.isOpened():
	print("Cannot open left cameras")
	exit()
if not cap_r.isOpened():
	print("Cannot open right cameras")
	exit()
while cap_l.isOpened() and cap_r.isOpened():
	# Capture frame-by-frame
	ret_l, frame_l = cap_l.read()
	ret_r, frame_r = cap_r.read()
	# if frame is read correctly ret is True
	if not ret_l:
		print("Can't receive frame from left camera (stream end?). Exiting ...")
		break
	if not ret_r:
		print("Can't receive frame from right camera (stream end?). Exiting ...")
		break

	# Display the captured frames
	cv.imshow("Original left frame", frame_l)
	cv.imshow("Original right frame", frame_r)
	cv.waitKey(100)

	# Camera calibration: https://learnopencv.com/camera-calibration-using-opencv/ and https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/

	# Correct images
	frame_l_corrected= cv.remap(frame_l, Left_Stereo_Map[0], Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
	frame_r_corrected= cv.remap(frame_r, Right_Stereo_Map[0], Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

	# Display the corrected frames
	cv.imshow("Corrected left frame", frame_l_corrected)
	cv.imshow("Corrected right frame", frame_r_corrected)
	cv.waitKey(0)
	
	# Our operations on the frame come here
	# frame_l = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
	# frame_r = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

	# write video to file
	# out.write(frame_l)

	# Anaglyph: https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/
	out = frame_r.copy()
	out[:,:,0] = frame_r[:,:,0]		# Copy Red data from Right frame
	out[:,:,1] = frame_r[:,:,1]		# Copy Green data from Right frame
	out[:,:,2] = frame_l[:,:,2]		# Copy Blue data from Left frame
	# Display the resulting Anaglyph frame
	cv.imshow("Anaglyph Output", out)
	
	key = cv.waitKey(0)
	if key == 115:
		# Pressed s
		# Save to file
		print("Save captured images")
		cv.imwrite("anaglyph.png", out)
	elif key == 113:
		# Pressed q
		break

# When everything done, release the capture
cap_l.release()
cap_r.release()
# out.release()
cv.destroyAllWindows()
