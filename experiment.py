#-------------------------------------------------------------------------------
# Name:
# Purpose:
#
# Author:      Admin
#
# Created:     08/08/2016
# Copyright:   (c) Admin 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import ImageSynthesis

import numpy as np
import scipy.io as sio
import matplotlib

from skimage import segmentation
from segment import clusters
from segment import regions
from measure import performance
from segment import mslic, features

def main():

	matplotlib.use('Agg')
	im1 = cv2.imread("data/0degree.bmp", cv2.IMREAD_COLOR)
	im2 = cv2.imread("data/15degree.bmp", cv2.IMREAD_COLOR)
	im3 = cv2.imread("data/30degree.bmp", cv2.IMREAD_COLOR)
	im4 = cv2.imread("data/45degree.bmp", cv2.IMREAD_COLOR)

	avg_im = ImageSynthesis.average_image(4, [im1, im2, im3, im4])
	max_im = ImageSynthesis.maximum_image(4, [im1, im2, im3, im4])

	gt = cv2.imread("data/gt.jpg", 0)
	thres = 3       # threshold of distance between edge pixel and g.t. edge pixel

	# Multi-channel SLIC, our method
	print ("begin multi-channel slic")
	img = [im1, im2, im3, im4]


	# load the pre-segmented image
	data = sio.loadmat("quartz_superpixel.mat")
	L = data["L"]


	L = clusters.merge_regions(img, L, gt, iteration = 50)
	bond_L = segmentation.find_boundaries(L)
	print (performance.boundary_recall(gt, bond_L, 3))
	print (performance.precision(gt, bond_L, 3))


if __name__ == '__main__':
	main()
