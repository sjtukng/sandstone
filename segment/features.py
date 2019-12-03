#-------------------------------------------------------------------------------
# Name:        features
# Purpose:     compute superpixel features and feature similarity
#
# Author:      ISet Group
#
# Created:     25/09/2016
#-------------------------------------------------------------------------------
import os
import cv2
import numpy as np
import tensorflow as tf
from math import sqrt
from skimage.filters import gabor
from skimage.filters import scharr
from keras.models import Model
from nn.PetroNetModel import get_model
from segment import regions

# extract the feature of the entire image
# input:    im, image to be segmented. (Note: im is a multi-angle image,
#               each angle is a single image in RGB color space)
# output:   fm, the feature maps of the input images
def nn_feature(im):
	# load the pre-trained model and network parameters
	model = get_model()
	model.load_weights('./model5_roi.h5')
    # we need the outputs after the last convolution layer (actually, the 2D pooling layer)
	model1 = Model(input=model.input, output=model.get_layer('pool2').output)
	fm = []
	number_of_angle = len(im)
	[h, w, c] = im[0].shape
	data = np.empty((number_of_angle, h, w, 1), dtype="float32")
	for i in range(number_of_angle):
		gray_img = cv2.cvtColor(im[i], cv2.COLOR_BGR2GRAY)
		data[i, :, :, 0] = gray_img
	fm = model1.predict(data, verbose = 0)
	return fm

# compute the feature of image region
# input:    im, the input image
# 			fm, the feature map of an image
#           L, labeled image of superpixels (range from 1 to k).
#           idx, the label index
# output:   ft, the feature of region idx
def feature_of_region(im, fm, L, idx):

	pos = np.where(L == idx)

	x_cordinates = pos[0]
	y_cordinates = pos[1]

	src_top = min(x_cordinates)
	src_left = min(y_cordinates)
	src_bottom = max(x_cordinates)
	src_right = max(y_cordinates)

	(h, w, c) = im.shape
	(H, W, C) = fm.shape

	# transform the coordinate from the original system to the feature map system
	dst_top = int(src_top * H / h)
	dst_left = int(src_left * W / w)
	dst_bottom = int(src_bottom * H / h)
	dst_right = int(src_right * W / w)

	feat_region = fm[dst_top:dst_bottom, dst_left:dst_right]

	ft = cv2.resize(feat_region,(5, 5))
	ft = np.reshape(ft, 400)

	return ft

# compute the similarity between two regions
# input:    rf1, the feature of region 1
# 			rf2, the feature of region 2
# output:   sim, the similarity between two region 1 and region 2
def region_similarity(rf1, rf2):
	sim = 0.0
	for i in range(len(rf1)):
		sim = sim + (rf1[i] - rf2[i]) * (rf1[i] - rf2[i])
	return sim


# normalize vector/matrix to [0,1]
def normalization(x):
	max = np.max(x)
	min = np.min(x)
	x = (x - min) / (max - min)
	return x

