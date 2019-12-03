#-------------------------------------------------------------------------------
# Name:			image
# Purpose:      a class for images
#
# Author:		ISet Group
#
# Created:		2016-06-03
#-------------------------------------------------------------------------------
import numpy as np
import os
import cv2
from skimage import feature

from options import Option
from common import files

# object definition for the Image
class Image(object):
	# ------ contructor ------------
	# input:
	#   imgFile, dir name of the image file
	def __init__(self, imgFile):
		# ------- the fields ---------
		self.name = files.get_file_names(imgFile)[0]
		self.rgb = cv2.imread(imgFile, cv2.IMREAD_COLOR)
		if not (self.rgb is None):
			gray = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)
			self.lbp = feature.local_binary_pattern(gray, Option.nPoints, \
					Option.radius, Option.lpbMeth)

	# get name of the image
	def get_name(self):
		return self.name

	# get rgb matrix
	def get_rgb(self):
		return self.rgb

	# get lbp matrix
	def get_lbp(self):
		return self.lbp

	# get gray matrix
	def get_gray(self):
		return cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)

	# get hsv matrix
	def get_hsv(self):
		return cv2.cvtColor(self.rgb, cv2.COLOR_BGR2HSV)

	# get lab matrix
	def get_lab(self):
		return cv2.cvtColor(self.rgb, cv2.COLOR_BGR2LAB)

	# check if the image is valid
	def is_valid(self):
		return not (self.rgb is None)
