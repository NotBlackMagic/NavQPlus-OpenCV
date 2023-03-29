import numpy as np
import cv2 as cv
import glob

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
cap_l = cv.VideoCapture('v4l2src device=/dev/video3 ! video/x-raw, framerate=30/1, width=640, height=480 ! appsink', cv.CAP_GSTREAMER)
cap_r = cv.VideoCapture('v4l2src device=/dev/video4 ! video/x-raw, framerate=30/1, width=640, height=480 ! appsink', cv.CAP_GSTREAMER)

i = 0
path_l = "./data/stereoL/"
path_r = "./data/stereoR/"

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

	key = cv.waitKey(0)
	print(key)
	if key == 115:
		# Pressed s
		# Save to file
		print("Save captured images")
		cv.imwrite(path_l + "img%d.png" % i, frame_l)
		cv.imwrite(path_r + "img%d.png" % i, frame_r)
		i += 1
	elif key == 113:
		# Pressed q
		break

# When everything done, release the capture
cap_l.release()
cap_r.release()
# out.release()
cv.destroyAllWindows()
