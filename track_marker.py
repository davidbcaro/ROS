import cv2
from markerfunctions import *
from imutils.video import VideoStream
from scipy.spatial import distance as dist
import numpy as np
import datetime
import argparse
import imutils
import time
import os
import math


SHAPE_RESIZE = 100.0
BLACK_THRESHOLD = 100
WHITE_THRESHOLD = 155
PATTERN = [0, 1, 0, 1, 0, 0, 0, 1, 1] 	
# PATTERN = [0, 1, 0, 1, 1, 1, 1, 0, 1] 
# PATTERN = [1, 0, 1, 0, 1, 0, 1, 0, 1] 
# PATTERN = [0, 1, 0, 1, 1, 1, 0, 1, 0] 

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

vs = VideoStream(usePiCamera=args["picamera"] > 0, resolution=(1024, 768)).start()
time.sleep(2.0)

 
while True:
	
	to = time.time()
	image = vs.read()
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5,5), 0)
	edges = cv2.Canny(gray, 100, 200)
	(_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
	
	for contour in contours:
		
		perimeter = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)
		      
		if len(approx) == 4:
			
			topdown_quad = get_topdown_quad(gray, approx.reshape(4, 2))
			
			if topdown_quad[(topdown_quad.shape[0]/100.0)*5,
							(topdown_quad.shape[1]/100.0)*5] > BLACK_THRESHOLD: continue
							
			glyph_found = False
			
			#for i in range(4):
			try:
				glyph_pattern = get_glyph_pattern(topdown_quad, BLACK_THRESHOLD, WHITE_THRESHOLD)
			except:
				continue
			#if glyph_pattern == PATTERN:
				#glyph_found = True
				#break
			if not glyph_pattern: continue
			
			if glyph_pattern == PATTERN:
				glyph_found = True
				break
			topdown_quad = rotate_image(topdown_quad, 90)
			
			if glyph_found:
				
				cv2.drawContours(image, [approx], -1, (0, 0, 255), 4)
				M = cv2.moments(approx)
				(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				cv2.line(image, (cX, cY), (cX, cY), (255, 0, 0), 3)
				width = 800
				Co = width/2
				cv2.line(image, (Co, Co), (Co, Co), (255, 0, 0), 3)
				cv2.line(image, (Co, Co), (Co, Co), (255, 0, 0), 3)
				cv2.line(image, (Co, Co), (cX, cY), (0, 255, 0), 3)
				
	cv2.imshow('Tracking', image)
	key = cv2.waitKey(1) & 0xFF
				
	if key == ord("q"):
		break
					
cv2.destroyAllWindows()
vs.stop()
