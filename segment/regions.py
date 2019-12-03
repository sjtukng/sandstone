#-------------------------------------------------------------------------------
# Name:        regions
# Purpose:     general functions for sub-regions
#
# Author:      ISet Group
#
# Created:     2016-06-03
#-------------------------------------------------------------------------------
import cv2
import numpy as np
import scipy.ndimage.measurements as snm

from skimage.segmentation import find_boundaries

# ensure labeled segments are continuous and connected
#   !!! all the other functions here assume L meet the requirements !!!
# input:    L, the label matrix identifying regions
# output:   L1, improved label matrix
def distinct_label(L):
	labels = np.unique(np.reshape(L,(L.shape[0]*L.shape[1],1)))
	L1 = np.zeros(L.shape, np.uint)
	curl = 0
	for l in labels:
		[bl, num] = snm.label(L == l)
		L1[bl>0] = bl[bl>0] + curl
		curl += num
	return L1


# find adjacency among regions use 4-connectedness
# input:    L, the labeling matrix (range from 1 to K)
# output:   A, the list of neighbors, indexed by labels
def adj_segs(L):
	num = np.max(L)
	A = []
	for k in range(num):
		A.append([])
	for r in range(L.shape[0]-1):
		for c in range(L.shape[1]-1):
			pairs=[(r,c+1),(r+1,c)]
			for v in pairs:
				if L[r,c]!=L[v] and L[r,c] not in A[L[v]-1]:
					A[L[r,c]-1].append(L[v])
					A[L[v]-1].append(L[r,c])
	return A

# compute adjacency matrix use 4-connectedness
# input:    L, the labeling matrix (range from 1 to K)
# output:   A, the adjacency matrix, 1 for adjacent, 0 for not adjacent
def adj_mat(L):
	num = np.max(L)
	A = np.zeros((num, num), np.uint8)

	for r in range(L.shape[0] - 1):
		for c in range(L.shape[1] - 1):
			pairs=[(r, c + 1),(r + 1, c)]
			for v in pairs:
				if L[r, c] != L[v]:
					A[L[r, c] - 1, L[v] - 1] = 1
					A[L[v] - 1, L[r, c] - 1] = 1
	return A

# find edge pixels between two superpixels
# input:    L, the labeling matrix (range from 1 to K)
#           x, label for superpixel 1
#           y, label for superpixel 2
# output:   edge, the matrix idetify the edge, where 1 denotes edge and 0 denotes non-edge
def edge_superpixels(L, x, y):
	bd = find_boundaries(L, mode = 'outer', background = x)
	e1 = bd * (L == y)
	bd2 = find_boundaries(L)
	e2 = bd2 * (L == y)
	edge = e2 - e1
	return edge


# compute centers of the regions as collections of color values and coordinates
# input:    img, the 3-channel image matrix
#           L, the label matrix identifying regions
# output:   G, num*6 (3-channel + x,y + number) array for the centers
def compute_centers(img, L):
	shape = L.shape
	num = np.max(L)
	G = np.zeros((num,6))

	for y in range(shape[0]):
		for x in range(shape[1]):
			if L[y,x] == 0:
				continue
			v = [img[y,x,0],img[y,x,1],img[y,x,2],x,y,1]
			G[L[y,x]-1,:] += v
	# Divide by number of pixels in each superpixel to get mean values
	for k in range(num):
		G[k,0:5] = np.round(G[k,0:5]/G[k,5],1)
	return G
