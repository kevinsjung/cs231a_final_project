# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from PIL import Image
import pandas as pd
import os, os.path
import csv
import re

def test():
	# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-p", "--shape-predictor", required=True,
	# 	help="path to facial landmark predictor")
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	# args = vars(ap.parse_args())

	path = 'Manually_Annotated_Images'

	first = ['ImageName', 'Gap', 'UpperLip', 'LowerLip', 'LipRatio']
	outputfilename = path + '/output.csv'
	with open(outputfilename, 'a') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(first)

	valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg"]

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	subpath = [x[0] for x in os.walk(path)]

	for sub in subpath:
		imgs = []
		images = []
		gap = []
		upperlip_distance = []
		lowerlip_distance = []
		badimages = []
		lipratio = []

		folder = re.findall('\d+', sub)

		badfilename = path + '/badimages.csv'

		print sub

		for f in os.listdir(sub):
			ext = os.path.splitext(f)[1]
			if ext.lower() not in valid_images:
				continue
			imgs.append(f)

		for im in imgs:
			# load the input image, resize it, and convert it to grayscale
			# image = cv2.imread(args["image"])
			image = cv2.imread(str(sub) + '/' + str(im))
			image = imutils.resize(image, width=500)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# detect faces in the grayscale image
			rects = detector(gray, 1)

			if not rects:
				with open(badfilename, 'a') as myfile:
					wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
					wr.writerow([str(sub)+'/'+str(im)])
			top_dists = []
			bottom_dists = []

			# loop over the face detections
			for (i, rect) in enumerate(rects):
				# determine the facial landmarks for the face region, then
				# convert the landmark (x, y)-coordinates to a NumPy array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# print shape
				for p in shape:
					cv2.circle(image, (int(p[0]), int(p[1])), 3, (255, 0, 0), -11)

				top = [48, 49, 50, 51, 52, 53, 54]
				bottom = [48, 59, 58, 57, 56, 55, 54]

				face_width = calculate_distance(shape, 0, 16)

				for j in range(1, 7):
					top_dists.append(calculate_distance(shape, top[j], top[j - 1]))
					bottom_dists.append(calculate_distance(shape, bottom[j], bottom[j - 1]))

				imfilename = str(folder[0]) + '/' + im
				localgap = calculate_distance(shape, 62, 66)/face_width
				localupper = np.average(top_dists)
				locallower = np.average(bottom_dists)
				localratio = locallower/localupper

				images.append(imfilename)
				gap.append(localgap)
				upperlip_distance.append(localupper)
				lowerlip_distance.append(locallower)
				lipratio.append(localratio)

				outline = [imfilename, localgap, localupper, locallower, localratio]

				outputfilename = path + '/output.csv'
				with open(outputfilename, 'a') as myfile:
					wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
					wr.writerow(outline)
				# cv2.line(image, (int(shape[48][0]), int(shape[48][1])), (int(shape[54][0]), int(shape[54][1])), (0, 0, 255))
				# cv2.circle(image, (int((shape[66][0] + shape[62][0]) / 2), int((shape[66][1] + shape[62][1])) / 2), 2, (0, 255, 0), -11)

				# print gap
				# print upperlip_distance
				# print lowerlip_distance
				# cv2.imshow('Test Imag', image)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()

					# loop over the face parts individually
		#
		# total_images.append(images)
		# total_gap.append(gap)
		# total_upperlip_distance.append(upperlip_distance)
		# total_lowerlip_distance.append(lowerlip_distance)
		# total_ratio.append(lipratio)

		# l = [(images), (gap), (upperlip_distance), (lowerlip_distance), (lipratio)]
		# list = zip(*l)
		# print list
		# #
		# first = ['ImageName', 'Gap', 'UpperLip', 'LowerLip', 'LipRatio']
		# filename = str(sub) + '/output.csv'
		# with open(filename, 'wb') as myfile:
		# 	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		# 	wr.writerow(first)
		# 	for li in list:
		# 		wr.writerow(li)



		# totals = [(total_images), (total_gap), (total_upperlip_distance), (total_lowerlip_distance), (total_ratio)]
		# tlist = zip(*totals)
		# print tlist
		# first = ['ImageName', 'Gap', 'UpperLip', 'LowerLip', 'LipRatio']
		# filename = path + '/output.csv'
		# with open(filename, 'a') as myfile:
		# 	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		# 	wr.writerow(first)
		# 	for li in tlist:
		# 		wr.writerow(li)

		#
		# for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# 	# clone the original image so we can draw on it, then
		# 	# display the name of the face part on the image
		# 	clone = image.copy()
		# 	cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		# 		0.7, (0, 0, 255), 2)
		#
		# 	# loop over the subset of facial landmarks, drawing the
		# 	# specific face part
		# 	for (x, y) in shape[i:j]:
		# 		cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
		#
		# 	# extract the ROI of the face region as a separate image
		# 	# (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		# 	# roi = image[y:y + h, x:x + w]
		# 	# roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		#
		# 	# show the particular face part
		# 	# cv2.imshow("ROI", roi)
		#
		# 	cv2.imshow("Image", clone)
		# 	cv2.waitKey(0)

		# visualize all facial landmarks with a transparent overlay
		# output = face_utils.visualize_facial_landmarks(image, shape)
		# cv2.imshow("Image", output)
		# cv2.waitKey(0)

def calculate_distance(pixels, point1, point2):
	return np.sqrt((pixels[point1][0]-pixels[point2][0])**2+(pixels[point1][1]-pixels[point2][1])**2)


if __name__ == '__main__':
	test()