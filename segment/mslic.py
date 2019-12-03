#--- ----------------------------------------------------------------------------
# Name:        mslic
# Purpose:     multi-channel slic
#
# Author:      ISet Group
#
# Created:     23/08/2016
#-------------------------------------------------------------------------------

import cv2
import math

import numpy as np
import scipy.stats as stats

from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from scipy.ndimage.measurements import label
from skimage.exposure import histogram
from scipy.stats import entropy
from segment import regions
from segment import features


# multi-channel slic
# input:    im, image to be segmented. (Note: im is a multi-channel image,
#               each channel is a single image in RGB color space)
#           k, number of desired superpixels.
#           m, Weighting factor between colour and spatial differences.
#           seRadius, Regions morphologically smaller than this are merged with
#                     adjacent regions. Try a value of 1 or 1.5. Use 0 to disable.
#           iteration, number of iteration
# output:   L, labeled image of superpixels (range from 0 to k - 1).

def mslic(im, k, m, seRadius, iteration):
	N = len(im)     # N is the number of image channel (NOT COLOR CHANNEL)
	if N > 1:
		[rows, cols, chan] = im[0].shape
	# convert image to CIElab color space
	for i in range(N):
		im[i] = cv2.cvtColor(im[i], cv2.COLOR_BGR2LAB)

	## initial segment, partiton the image into k superpixels
	# spacing between grid elements assuming square grid
	S = math.sqrt(rows * cols / k);
	# number of nodes per row
	nodeCols = round(cols / S);
	S = cols / nodeCols;    # Recompute S
	# number of rows of nodes
	nodeRows = round(rows / S);
	vSpacing = rows / nodeRows;
    # recompute k
	k = nodeRows * nodeCols
	# allocate memory and initialise clusters, labels and distances.
	C = np.zeros((k, 2), np.uint16)     # Cluster center
	L = -np.ones((rows, cols), np.int)			# Pixel labels.
	d = np.inf * np.ones((rows, cols))	# Pixel distances from cluster centres.
	print ("Initialise cluster centers")
    # initialise cluster centers
	kk = 0
	for ri in range(int(nodeRows)):
		for ci in range(int(nodeCols)):
			C[kk, 0] = ri * S + S / 2
			C[kk, 1] = ci * vSpacing + vSpacing / 2
##			print 'C[' + str(kk) + ']=' + str(C[kk, 0]) + ',' + str(C[kk, 1])
			kk = kk + 1
	# Move cluster centers to loweset gradient position in 3*3 neighborhood among all channels
	print ("perturb cluster centers")
	ig = np.zeros((N, 9), np.float)
	# 0   1   2
	# 3   4   5
	# 6   7   8
	for i in range(int(k)):
		for j in range(N):
			ig[j, 0] = image_gradient(im[j], C[i, 0] - 1, C[i, 1] - 1)
			ig[j, 1] = image_gradient(im[j], C[i, 0] - 1, C[i, 1])
			ig[j, 2] = image_gradient(im[j], C[i, 0] - 1, C[i, 1] + 1)
			ig[j, 3] = image_gradient(im[j], C[i, 0], C[i, 1] - 1)
			ig[j, 4] = image_gradient(im[j], C[i, 0], C[i, 1])
			ig[j, 5] = image_gradient(im[j], C[i, 0], C[i, 1] + 1)
			ig[j, 6] = image_gradient(im[j], C[i, 0] + 1, C[i, 1] - 1)
			ig[j, 7] = image_gradient(im[j], C[i, 0] + 1, C[i, 1])
			ig[j, 8] = image_gradient(im[j], C[i, 0] + 1, C[i, 1] + 1)
		pos = np.argmin(ig) % 9
		if pos == 0:
			C[i, 0] -= 1
			C[i, 1] -= 1
		elif pos == 1:
			C[i, 0] -= 1
		elif pos == 2:
			C[i, 0] -= 1
			C[i, 1] += 1
		elif pos == 3:
			C[i, 1] -= 1
		elif pos == 5:
			C[i, 1] += 1
		elif pos == 6:
			C[i, 0] += 1
			C[i, 1] -= 1
		elif pos == 7:
			C[i, 0] += 1
		elif pos == 8:
			C[i, 0] += 1
			C[i, 1] += 1

	## begin iteration
	S = int(S)
	for it in range(iteration):
		print ("iteration " + str(it + 1) + "...")
		for kk in range(int(k)):     # for each cluster
			# get subimage around the cluster center
			rmin = max(C[kk, 0] - S, 0)
			rmax = min(C[kk, 0] + S, rows - 1)
			cmin = max(C[kk, 1] - S, 0)
			cmax = min(C[kk, 1] + S, cols - 1)
##			print "cluster " + str(kk) + ":rmin=" + str(rmin) + ",rmax=" \
##				+ str(rmax) + ",cmin=" + str(cmin) + ",cmax=" + str(cmax)
			subim = np.zeros((N, rmax-rmin+1, cmax-cmin+1, 3), np.uint8)
			for j in range(N):
				subim[j] = im[j][rmin:rmax+1, cmin:cmax+1]
			# compute distances between C(kk) and subimage
			D = dist(C[kk], subim, rmin, cmin, S, m)
			# if any pixel distance from the cluster center is less than its previous value update its distance and label
			subd = d[rmin:rmax+1, cmin:cmax+1]
			subl = L[rmin:rmax+1, cmin:cmax+1]
			updateMask = D < subd
			subd[updateMask] = D[updateMask]
			subl[updateMask] = kk
			d[rmin:rmax+1, cmin:cmax+1] = subd
			L[rmin:rmax+1, cmin:cmax+1] = subl
		# update cluster centers with mean values
		C = np.zeros((k, 2), np.uint16)	# Reset cluster center data, 0:1 is row, col of center
		for kk in range(int(k)):
			pos = np.where(L == kk)
			num = len(pos[0])
			C[kk, 0] = sum(pos[0]) / num
			C[kk, 1] = sum(pos[1]) / num
	return L

# compute distances between cluster center and subimage
# Distance = sqrt( dc^2 + (ds/S)^2*m^2 )
# input:    C, Cluster being considered
#			lab, sub-image surrounding cluster centre
#               (Note: im is a multi-channel image, each channel is a single image in CIElab color space)
#			r1,c1, row and column of top left corner of sub image within the overall image.
#			S, grid spacing
#			m, weighting factor between colour and spatial differences.
# output:   D, Distance image giving distance of every pixel in the subimage from the cluster center
def  dist(C, lab, r1, c1, S, m):
	N = len(lab)
	if N > 1:
		[rows, cols, chan] = lab[0].shape
	D = np.zeros((rows, cols), np.float)

	# manually construct a matrix for computing color differences
	center_lab = np.ones([N, rows, cols, 3], np.int)
	for n in range(N):
		center_l = lab[n, C[0]-r1, C[1]-c1, 0]
		center_a = lab[n, C[0]-r1, C[1]-c1, 1]
		center_b = lab[n, C[0]-r1, C[1]-c1, 2]
		center_lab[n, :, :, 0] *= center_l
		center_lab[n, :, :, 1] *= center_a
		center_lab[n, :, :, 2] *= center_b

	dc2 = np.zeros((N, rows, cols), np.float)
	dc2 = ((lab - center_lab) ** 2).sum(axis=3)

    # manually construct a matrix for computing spatial differences
	img_xy = np.zeros([rows, cols, 2], np.int)
	for i in range(rows):
		for j in range(cols):
			img_xy[i, j, 0] = r1 + i
			img_xy[i, j, 1] = c1 + j
	ds2 = (img_xy[:,:,0] - C[0]) ** 2 + (img_xy[:,:,1] - C[1]) ** 2

	d = np.zeros((N, rows, cols), np.float)
	for n in range(N):
		d[n] = dc2[n] + ds2 / S * (m ** 2)
	D = d.max(axis = 0)

	return D

# compute image gradient
# input:    lab, the 3-channel color matrix for image (CIElab color space)
#           x, the x-coordinate for pixel
#           y, the y-coordinate for pixel
# output:   g, the image gradient for pixel (x, y)
def image_gradient(lab, x, y):
	rows = lab.shape[0]
	cols = lab.shape[1]

	if x <= 0 or x >= rows-1 or y <= 0 or y >= cols-1:
		return np.inf

	p1 = lab[x+1, y]
	p2 = lab[x-1, y]
	p3 = lab[x, y+1]
	p4 = lab[x, y-1]
	g = (int(p1[0]) - int(p2[0])) ** 2 + (int(p1[1]) - int(p2[1])) ** 2 + (int(p1[2]) - int(p2[2])) ** 2 \
	  + (int(p3[0]) - int(p4[0])) ** 2 + (int(p3[1]) - int(p4[1])) ** 2 + (int(p3[2]) - int(p4[2])) ** 2
	return g

### merge similar regions
### input:    im, image to be segmented. (Note: im is a multi-channel image,
###               each channel is a single image in RGB color space)
###           L, labeled image of superpixels (range from 1 to k).
### output:   L, the labeled image after region merging
##
##def merge_regions(im, L, iteration = 50):
##
##	print "Merge similar regions using RAG"
##
##	N = len(im)     # N is the number of image channel (NOT COLOR CHANNEL)
##	if N > 1:
##		[rows, cols, chan] = im[0].shape
##
##	rag = regions.adj_mat(L)	# rag is adjacent matrix
##
##	K = np.max(L)				# number of superpixels
##	delta = np.inf * np.ones((K, K), np.float)  # the similarity matrix, initilized as infinity
##
##	for i in range(K):      # i in range [0:K-1]
##		for j in range(i + 1, K):    # j in range [i+1:K-1], use upper-triangle matrix only
##			# constuct the graph node, i.e. [index, similarity]
##			# note: the edge (x, y) should satisfy x <= y
##			if rag[i, j] == 0:    # if superpixel i and j are not adjacent
##				continue
##			sim = mc_region_similarity(im, L, i + 1, j + 1)     # label index start from 1
##			delta[i, j] = sim     # store weight of edge to matrix
##
##	for it in range(iteration):
##		print "iteration " + str(it + 1)
##		# find the minimum weight position, x < y
##		pos = np.argmin(delta)
##		x = pos / K
##		y = pos % K
##
##		# update L
##		L[L == y + 1] = x + 1
##
##		# update RAG
##		# add y's neighbors to x's neighbor list
##		ny = (np.nonzero(rag[y, :]))
##		for i in range(len(ny)):
##			rag[x, ny[i]] = 1
##			rag[ny[i], x] = 1
##
##		# empty y's neighbor list
##		rag[:, y] = 0
##		rag[y, :] = 0
##
##		# update delta
##		delta[:, y] = np.inf
##		delta[y, :] = np.inf
##
##		for i in range(K):
##			if rag[i, x] == 1 and i < x:
##				delta[i, x] = mc_region_similarity(im, L, i + 1, x + 1)     # label index start from 1
##
##		for j in range(K):
##			if rag[x, j] == 1 and x < j:
##				delta[x, j] = mc_region_similarity(im, L, x + 1, j + 1)     # label index start from 1
##
##	return L

### compute histogram of image region
### input:    im, the multi-channel image, each channel is a single image in RGB color space
###           L, labeled image of superpixels (range from 1 to k).
###           idx, the label index
### output:   hist, histogram of region idx (include multi-channel histogram)
##def mc_hist(im, L, idx):
##
##	N = len(im)     # N is the number of image channel (NOT COLOR CHANNEL)
##	if N > 1:
##		[rows, cols, chan] = im[0].shape
##
##	pos = np.where(L == idx)
##	num_of_pixels = len(pos[0])
##
##	## convert rgb image to gray-level image
##	gray_im = np.zeros((N, rows, cols), np.uint8)
##	for i in range(N):
##		gray_im[i] = cv2.cvtColor(im[i], cv2.COLOR_BGR2GRAY)
##
##	hist = np.zeros(N ,list)
##	for n in range(N):
##		l = []
##		for i in range(num_of_pixels):
##			l.append(gray_im[n, pos[0][i], pos[1][i]])
##		hist[n], _ = np.histogram(np.array(l), range(0,257,16))
##
##	return hist
##
### compute similarity of two histograms, using Jensen-Shannon divergence
### input:    h1, histogram of region 1 (multi-channel)
###           h2, histogram of region 2 (multi-channel)
### output: sim, the similarity of histograms 1 and region 2
##def mc_hist_similarity(h1, h2):
##	N = len(h1)
##	s = np.zeros(N, np.float)
##	for n in range(N):
##		s1 = entropy(h1[n], h2[n])
##		s2 = entropy(h2[n], h1[n])
##		s[n] = 0.5 * (s1 + s2)
##	sim = np.max(s)
##	return sim
##
### compute similarity of two regions
### input:    im, the multi-channel image, each channel is a single image in RGB color space
###           L, labeled image of superpixels (range from 1 to k).
###           i, the region 1's label
###           j, the region 2's label
### output: sim, the similarity of region 1 and region 2
##def mc_region_similarity(im, L, i, j):
##	hist_i = mc_hist(im, L, i)
##	hist_j = mc_hist(im, L, j)
##	sim = mc_hist_similarity(hist_i + 0.1, hist_j + 0.1)    # "+ 0.1" to avoid zeros in histogram
##	# region area should be considered in the similarity metric
##	num_superpixel_i = np.sum(L == i)	# number of superpixel i + 1
##	num_superpixel_j = np.sum(L == j)	# number of superpixel nb[j]
##	sim = float(num_superpixel_i * num_superpixel_j) / (num_superpixel_i + num_superpixel_j) * sim
##	return sim
