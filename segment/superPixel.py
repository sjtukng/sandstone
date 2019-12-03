#-------------------------------------------------------------------------------
# Name:        superPixel
# Purpose:     general interface for super-pixel functions
#
# Author:      ISet Group
#
# Created:     2016-04-03
#-------------------------------------------------------------------------------
import cv2
import numpy as np
from skimage import segmentation

import regions
from options import Option
from images import Image
from common.unionFind import UnionFinder
from segment import dbscan
from segment import clusters

# generate superpixels by clustering methods
# input:    image, the image object
#           algorithm, the superpixel algorithm, i.e. slic, meanshift, turbopixels ...
#           tsize, tiny regions no larger than tsize will be merged
# output:   L, the label matrix identifying the regions
def generate_superpixels(image, algorithm):
	if algorithm == 'slic':
		print 'call the base slic algorithm'
		L = segmentation.slic(image.get_rgb(), n_segments=Option.sNseg, compactness=Option.sCompact, \
				sigma=Option.sSigma,convert2lab=True)   #call slic
	elif algorithm == 'quickshift':
		print 'call the quickshift algorithm'
		L = segmentation.quickshift(image.get_rgb(), ratio=Option.qRatio, kernel_size=Option.qKernel, \
				max_dist=Option.qDist, return_tree=False, sigma=0, convert2lab=True, random_seed=Option.qSeed)
	else:
		print 'call the felzenszwalb algorithm'
		L = segmentation.felzenszwalb(image.get_rgb(), scale = 20, sigma = 5, min_size = 60)

	L = regions.distinct_label(L)
	print np.max(L),'superPixels generated'
	L = clusters.merge_tiny_regions(image.get_lab(), L, Option.tsize)
	print np.max(L),'superPixels remained'

	#clustering the super-pixels
	print 'merge the superpixels'
	L = dbscan.merge_region(image, L, [Option.dbTresh, Option.dbText])

	return L
