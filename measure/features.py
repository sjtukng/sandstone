#-------------------------------------------------------------------------------
# Name:        features
# Purpose:     extract features from image
#
# Author:      ISet Group
#
# Created:     2016-04-03
#-------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
from skimage import exposure

# object definition for Feature
class Feature(object):
	# ------ contructor ------------
	# input: dv, a list of pixels as singular values
	def __init__(self, dv):
		# @dv is a 1-D list
		numpixel = len(dv)
		# number of pixels
		self.count = numpixel
		# compute histogram
		bins = np.arange(256)
		minVal = np.min(dv)
		factor = 1.0
		if np.max(dv)-minVal > 0:
			factor = 255.0/(np.max(dv)-minVal)
		hist,bins = np.histogram((dv-minVal)*factor, bins)
		# compute mean value
		self.mean = np.mean(dv)
		# compute variance
		self.var = np.var(dv)
		# compute mode
		self.mode = list(hist).index(np.max(hist))/factor
		# compute range
		self.range = np.max(dv) - np.min(dv)
		sortList = sorted(dv)
		# compute median
		if numpixel % 2 == 1:
			self.median = sortList[int(numpixel/2)]
		else:
			self.median = (1.0*sortList[int(numpixel/2-1)]+sortList[int(numpixel/2)]) / 2.0
		# compute quartRange
		one_four = int(numpixel / 4)
		if numpixel % 4 == 0:
			self.quartRange = sortList[3*one_four-1] - sortList[one_four-1]
		else:
			q = numpixel/4.0
			q_one = (q-one_four)*sortList[one_four] + (one_four-q+1)*sortList[one_four-1]
			q_three = 3.0*(q-one_four)*sortList[3*one_four] + (3.0*(one_four-q)+1.0)*sortList[3*one_four-1]
			self.quartRange = q_three - q_one
		# compute smoothness
		self.smoothness = 1.0 - 1.0/(1+self.var)
		# compute probability array
		p = hist / float(numpixel)
		# compute uniformity
		self.uniformity = np.sum(p * p)
		# compute entropy
		self.entropy = -np.sum(p * np.log2(p+1.0e-10))
		# compute mean absolute deviation
		self.mad = np.sum(np.fabs(dv - self.mean)) / float(numpixel)
		# compute skewness
		self.skewness = stats.skew(dv)
		# compute kurtosis
		self.kurtosis = stats.kurtosis(dv)

	# get the feature vector
	# output: the feature vector
	def get_fv(self):
		fv = [self.mean, self.var, self.mode, self.range, self.median, \
				self.quartRange, self.smoothness, self.uniformity, \
				self.entropy, self.mad, self.skewness, self.kurtosis]
		return fv


# compute features for a region represented as a vector list
# input:	vlist, a list of meta-value vectors
# output:   fv, a vector of computed features as list
def meta2fv(vlist):
	fv = []
	mtx = np.asarray(vlist)
	for idx in range(mtx.shape[1]):
		feature = Feature(mtx[:,idx])
		fv+= feature.get_fv()

	return fv


# compute features for a region represented as a matrix
# input:    mtx, a matrix containing the image data
# output:   fv, a vector of computed features as a list
def mtx2fv(mtx):
	shape = mtx.shape
	mlist = []
	# row*col*channels or row*col
	if len(shape) == 3:
		for idx in range(shape[2]):
			mlist.append(mtx[:,:,idx])
	else:
		mlist.append(mtx)

	fv = []
	for M in mlist:
		dv = []
		for idx in range(shape[0]):
			dv.extend(list(M[idx]))
		feature = Feature(dv)
		fv += feature.get_fv()

	return fv


# compute features from the lbp representation of an image
# input:    lbp, the matrix containing the lbp representation
# output:   a vector of lbp related features
def lbp2fv(lbp):
	nBins = 50
	hist, _ = exposure.histogram(lbp, nBins)

	return hist
