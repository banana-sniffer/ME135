# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from graphics import *
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,draw,show,figure

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# defines the lower/upper bounds of the ping pong ball
# in a HSV color space, as well as a list of tracked 
# points
orange1 = (30,255,255)
orange2 = (10,100,150)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# CURRENTLY NOT USED, WILL BE USING OUR WEBCAM
# FOR REAL TIME
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# set clock time 
t = 0

# initialize velocity for prediction
x_pos = []
t_between_frames = []
curr_frame = 0
num_frames_between = 5


# run the computer vision
while True:
	# grab the current frame
	frame = vs.read()

	# grab the current tick count
	e1 = cv2.getTickCount()

	# handle the frame received from the camera
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# create a mask for the ping pong ball, while removing
	# some dark spots to make a full circle
	mask = cv2.inRange(hsv, orange2, orange1)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# need to have one countour to run 
	if len(cnts) > 0:
		# PREVIOUSLY USED FOR ANIMATION IN SEPARATE WINDOW
		# for item in win.items[:]:
		# 	item.undraw()
		# win.update()

		# uses the largest contour to create a circle around it
		# as well as find the center
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		
		# ping pong ball must have a radius of at least 15 to be considered
		if radius > 15:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(255, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

			# draw the line under to simulate our platform
			# will grow and shrink to represent the depth of 
			# the platform
			cv2.rectangle(frame,(int(x)-(int(radius)+5),325),
				(int(x)+(int(radius)+5),325),(0,255,0),3)
			# when the ball is within a radius of the platform
			# simulates the platform rising up and hitting the 
			# ball
			if int(y) > 325 - int(radius):
				cv2.rectangle(frame,(int(x)-(int(radius)+5),300),
				(int(x)+(int(radius)+5),300),(255,0,0),3)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)-5):
		# igore the tracked points that are None
		if pts[i - 1] is None or pts[i] is None:
			continue

		# draw the line that is created that shows the movement
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# calculates the time between each tick as well
	# as the amount of total time that has elapsed
	e2 = cv2.getTickCount()
	t = t + (e2 - e1)/ cv2.getTickFrequency()
	if (curr_frame == num_frames_between):
		x_pos.append(int(x))
		t_between_frames.append(t)
		vel = (x_pos[1] - x_pos[0])/(t_between_frames[1]-t_between_frames[0])
		# shows us the velocity of the ball in the x-direction
		# over a time of 5 frames, using kinematics will be in 
		# units of pixels over time, later converted to inches
		# in need of finding the necessary conversion 
		print("Velocity"+str(vel))
		x_pos = []
		t_between_frames= []
		curr_frame = 0
	elif (curr_frame == 0):
		x_pos.append(int(x))
		t_between_frames.append(t)
		curr_frame = curr_frame + 1
	else:
		curr_frame = curr_frame + 1

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

	# if the 'c' key is pressed, clear the plot
	# CURRENTLY NOT IN USE, SLOWS DOWN PROGRAM 
	# TOO MUCH WILL DON'T NEED TO COLLECT AS 
	# MUCH DATA AS PREVIOUSLY THOUGHT
	if key == ord("c"):
		x_vals = []
		y_vals = []
		radii = []
		plt.plot(x_vals,y_vals)

# PREVIOUSLY USED TO GET 100 POINTS TO SHOW 
# POSITION OF THE BALL AS IT MOVES
# shows plot for 100 points
# plt.plot(x_vals,y_vals)
# plt.show()

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()