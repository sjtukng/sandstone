#-------------------------------------------------------------------------------
# Name:        dbscan
# Purpose:     implement the dbscan method of clustering regions
#
# Author:      ISet Group
#
# Created:     2016-06-08
#-------------------------------------------------------------------------------
import cv2
import numpy as np
from measure import features
from common import unionFind
from common import vectors
from segment import regions


# perform dbscan clustering of superpixels
# input:	image, the image object
#			L, the label matrix identifying regions
#           thresh, list of thresholds for matching adjacent superpixels
#			A, the adjacency matrix, default None
# output:   L1, the improved label matrix
def merge_region(image, L, thresh, A = None):
	if A is None:
		A = regions.adj_segs(L)
	num = np.max(L)
    # keep track of superpixels that have been visited
	visited = np.zeros(num, np.uint16)
	# keep track of clusters
	C = np.zeros(num, np.uint16)
	nc = 0
	# get statistics from the image
	S = get_statistics(image, L)
	# iterate among the regions
	for idx in range(num):
		if visited[idx] == 1:
			continue
		visited[idx] = 1
		# form a cluster
		nc = nc + 1
		C[idx] = nc
		# merge iteratively the adjacent neighbours
		fn = fit_neighbors(S, idx+1, A[idx], thresh)
		while (fn):
			k = fn.pop()
			if visited[k-1] == 1:
				continue
			visited[k-1] = 1
			C[k-1] = nc
			# get adjacent neighbors iteratively
			fn.extend(fit_neighbors(S, k, A[k-1], thresh))
	# relabel
	L1 = np.zeros(L.shape, np.uint16)
	for idx in range(num):
		L1[L==idx+1] = C[idx]
	print (nc, 'regions remained after clustering')
	return L1


# get statistics from the image for dbscan, may involve options
# inputs:   image, the image object
#           L, the label matrix for regions
# output:   the statistics for the regions
def get_statistics(image, L):
	lab = image.get_lab()
	G = regions.compute_centers(lab, L)
	lbp = image.get_lbp()
	# list of histgrams
	H = []
	for idx in range(np.max(L)):
		H.append(features.lbp2fv(lbp[L==idx+1]))
	return [G, H]


# Find indices of superpixels adjacent to the target with
# distances less than threshold.
# input:	S, the list of region statistics
#			idx, the index of the target superpixel, start from 1
#			N, list of its neighbors
#			thresh, the difference thresholds, factors of the deviation
# output:	fn, list of superpixels for merge
def fit_neighbors(S, idx, N, thresh):
	fn = []
	G = S[0]
	H = S[1]
	eth = thresh[0]
	for k in N:
		# compute distance
		d1 = vectors.euclid_dist(G[idx-1,0:3], G[k-1,0:3])
		d2 = vectors.jensen_shannon_divergence(H[idx-1], H[k-1])
		if (d1 < eth*eth and d2 < thresh[1]):
			fn.append(k)

	return fn
