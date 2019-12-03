#-------------------------------------------------------------------------------
# Name:        clusters
# Purpose:     general functions for clusters of regions
#
# Author:      ISet Group
#
# Created:     2016-06-05
#-------------------------------------------------------------------------------
import cv2
import csv

import numpy as np

from common import unionFind
from common import vectors
from segment import features, regions
from measure import performance
from skimage import segmentation
from nn.PetroNetModel import get_model


# merge the tiny superpixels
# input:	img, the 3-channal color image
#           L, labels for the superpixels
#           size, threshold for the tiny regions
# output:   L1, the improved labeling matrix without tiny superpixels
def merge_tiny_regions(img, L, size):
	print ("Merge tiny regions")
	ok = False
	while not ok:
		ok = True
		A = regions.adj_segs(L)
		G = regions.compute_centers(img, L)
		num = np.max(L)
		finder = unionFind.UnionFinder(num)
		cnt = 0
		for k in range(num):
			if G[k,5] > size or len(A[k]) == 0:
				continue
			cnt += 1
			ls = A[k]
			dv = np.zeros(len(ls))
			for j in range(len(ls)):
				dv[j] = vectors.euclid_dist(G[k,0:3], G[ls[j]-1,0:3])
			j = np.argmin(dv)
			finder.merge(k+1, ls[j])
			# re-arrange labels
		if cnt > 0:
			ok = False
			L1 = np.zeros(L.shape, np.uint16)
			kv = finder.get_kv()
			for k in range(1, num+1):
				L1[L==k] = kv[k]
			print (cnt, 'tiny super-pixels merged')
			L = L1
		else:
			L1 = L
	return L1


# merge similar regions
# input:    im, image to be segmented. (Note: im is a multi-angle image,
#               each angle is a single image in RGB color space)
#           L, labeled image of superpixels (range from 1 to k).
# output:   L1, the labeled image after region merging
# the similarity measurement is based on brightness, texture (gabor filter bank) and edge (scharr)
def merge_regions(im, L, gt, iteration = 50):
	print ("Merge similar regions using RAG")
	N = len(im)     							# N is the number of image angle (NOT COLOR CHANNEL)
	[h, w, c] = im[0].shape						# compute the shape of the input image
	feature_map = features.nn_feature(im)		# compute the feature map of the input images
	[H, W, C] = feature_map[0].shape			# compute the shape of the feature map

	rag = regions.adj_mat(L)		# rag is adjacent matrix
	K = np.max(L)					# number of superpixels

	sim = []  										# the similarity matrix (multi-angle)
	delta = np.inf * np.ones((K, K), np.float)  	# the final similarity matrix, initilized as infinity

	region_feature_vector = []

	for i in range(N):
		region_feature_vector.append([])
		sim.append(np.inf * np.ones((K, K), np.float))

	for i in range(N):
		for k in range(K):
			region_feature_vector[i].append(features.feature_of_region(im[i], feature_map[i], L, k + 1))

	L1 = np.array(L)
	for n in range(N):
		print ("process angle " + str(n+1))
		for i in range(K):					# i in range [0:K-1]
			if (i+1) % 50 == 0:
				print ("process " + str(i+1) + " superpixels")
			for j in range(i + 1, K):		# j in range [i+1:K-1], use upper-triangle matrix only
				# constuct the graph node, i.e. [index, similarity]
				# note: the edge (x, y) should satisfy x <= y
				if rag[i, j] == 0:			# if superpixel i and j are not adjacent
					continue

				sim[n][i, j] = features.region_similarity(region_feature_vector[n][i], region_feature_vector[n][j])


	for i in range(K):
		for j in range(i + 1, K):
			if rag[i, j] == 0:
				continue

			temp = []

			for n in range(N):
				temp.append(sim[n][i, j])
				temp.append(sim[n][i, j])
				temp.append(sim[n][i, j])
			delta[i, j] = max(temp)

	finder = unionFind.UnionFinder(K)

	# write similarity for analysis
	# outfile = file('iteration_slic.csv', 'wb')
	# writer = csv.writer(outfile)

	for it in range(iteration):
		print ("iteration " + str(it + 1))
		# find the minimum weight position, x < y
		pos = np.argmin(delta)
		x = int(pos / K)
		y = int(pos % K)
		# merge superpixel y+1 to superpixel x+1
##		print "merge superpixel " + str(y + 1) + " to superpixel " + str(x + 1)
		# update L1
		x1 = finder.find(x + 1)     # find x's root
		y1 = finder.find(y + 1)     # find y's root
		if x1 != y1:
			finder.merge(x1, y1)

		L1[L1 == (y + 1)] = (x + 1)		# merge y to x
		delta[x, y] = np.inf
		rag[x, y] = 0
		rag[y, x] = 0

		# update x's neighbor
		for i in range(K):
			if rag[i, x] == 1:



				b_s = compute_brightness_similarity(N, gray_im, L1, i+1, x+1, b_max_sim, b_min_sim)
				t_s = compute_texture_similarity(N, t_im, L1, i+1, x+1, t_max_sim, t_min_sim)
				e_s = compute_edge_similarity(N, e_im, L1, i+1, x+1, e_max_sim, e_min_sim)

				s = b_s + t_s + e_s
				if i < x:
					b_delta[i, x] = b_s
					t_delta[i, x] = t_s
					e_delta[i, x] = e_s
					delta[i, x] = s
				else:
					b_delta[x, i] = b_s
					t_delta[x, i] = t_s
					e_delta[x, i] = e_s
					delta[x, i] = s

		# update y's neighbor
		# note: y has been merged to x
		for i in range(K):
			if rag[i, y] == 1:
				b_s = compute_brightness_similarity(N, gray_im, L1, i+1, x+1, b_max_sim, b_min_sim)
				t_s = compute_texture_similarity(N, t_im, L1, i+1, x+1, t_max_sim, t_min_sim)
				e_s = compute_edge_similarity(N, e_im, L1, i+1, x+1, e_max_sim, e_min_sim)

				s = b_s + t_s + e_s
				if i < x:
					b_delta[i, x] = b_s
					t_delta[i, x] = t_s
					e_delta[i, x] = e_s
					delta[i, x] = s
					delta[i, x] = s
				else:
					b_delta[x, i] = b_s
					t_delta[x, i] = t_s
					e_delta[x, i] = e_s
					delta[x, i] = s
					delta[x, i] = s
				rag[i, y] = 0
				rag[y, i] = 0
				rag[i, x] = 1
				rag[x, i] = 1
				if i < y:
					delta[i, y] = np.inf
				else:
					delta[y, i] = np.inf

##		inter = Inter(K, it+1, rag, b_delta, t_delta, e_delta)
##		print "inter metric=" + str(inter)
##		intra = Intra(K, it+1, rag0, finder, b_delta0, t_delta0, e_delta0)
##		print "intra metric=" + str(intra)

		thres = 3
		bond_L = segmentation.find_boundaries(L1, mode='thick')
		br = performance.boundary_recall(gt, bond_L, thres)
		bp = performance.precision(gt, bond_L, thres)
		print ("recall=" + str(br))
		print ("precision=" + str(bp))
		print ("F-measure=" + str(2*br*bp/(br+bp)))

		# row = ['iteration ' + str(it+1), br, bp, 2 * br * bp / (br + bp)]
		# writer.writerow(row)

	# outfile.close()
	L1 = regions.distinct_label(L1)
	print (np.max(L1))
	return L1


# compute brightness similarity between two adjacent regions
def compute_brightness_similarity(N, gray_im, L, i, j, b_max, b_min):
	b_sim_vec = []
	for n in range(N):
		bf1 = features.b_feature(gray_im[n], L, i)
		bf2 = features.b_feature(gray_im[n], L, j)
		b_sim = features.b_similarity(bf1, bf2)             # brightness similarity
		b_sim = (b_sim - b_min) / (b_max - b_min)
		b_sim_vec.append(b_sim)
	return max(b_sim_vec)


# compute texture similarity between two adjacent regions
def compute_texture_similarity(N, t_im, L, i, j, t_max, t_min):
	t_sim_vec = []
	for n in range(N):
		tf1 = features.t_feature(t_im[n], L, i)
		tf2 = features.t_feature(t_im[n], L, j)
		t_sim = features.t_similarity(tf1, tf2)             # texture similarity
		t_sim = (t_sim - t_min) / (t_max - t_min)
		t_sim_vec.append(t_sim)
	return max(t_sim_vec)

# compute edge similarity between two adjacent regions
def compute_edge_similarity(N, e_im, L, i, j, e_max, e_min):
	e_sim_vec = []
	for n in range(N):
		e_sim = features.e_similarity(e_im[n], L, i, j)      	# edge similarity
		e_sim = (e_sim - e_min) / (e_max - e_min)
		e_sim_vec.append(e_sim)
	return max(e_sim_vec)

# compute Intra-region homogeneity of segmentation
# input:    Ns, the initial number of superpixels after MSLIC
#           k, iteration number
#           rag0, the RAG of initial segmentation
#           finder, the union-finder
#           b_delta0, the initial brightness similarity matrix
#           t_delta0, the initial texture similarity matrix
#           e_delta0, the initial edge similarity matrix
# output:   F, the Intra-region homogeneity
def Intra(Ns, k, rag0, finder, b_delta0, t_delta0, e_delta0):

	F = 0
	for i in range(Ns):
		sl = finder.list[i+1]       # sl is the initial superpixels included in superpixel i+1 after k-th merging
		num = len(sl)
		if num == 0 or num == 1:
			continue
		temp = 0
		count = 0
		for m in range(num):
			for n in range(m+1, num):
				if rag0[sl[m]-1, sl[n]-1] == 0:      # superpixels m and n are not adjacent
					continue
				p = sl[m] - 1
				q = sl[n] - 1
				t = 0
				if p > q:
					t = p
					p = q
					q = t

				b_temp = b_delta0[p, q]
				t_temp = t_delta0[p, q]
				e_temp = e_delta0[p, q]
				temp = temp + b_temp + t_temp + e_temp
				count += 1
		temp = temp / count
		F += temp
	return F


# compute Inter-region homogeneity of segmentation
# input:    Ns, the initial number of superpixels after MSLIC
#           k, iteration number
#           rag, the RAG of segmentation
#           b_delta, brightness similarity matrix
#           t_delta, texture similarity matrix
#           e_delta, edge similarity matrix
# output:   F, the Inter-region homogeneity
def Inter(Ns, k, rag, b_delta, t_delta, e_delta):
	F = 0
	for i in range(Ns):
		for j in range(i+1, Ns):
			if rag[i, j] == 0:  # i and j are not adjacent
				continue
			b_sim = b_delta[i, j]
			t_sim = t_delta[i, j]
			e_sim = e_delta[i, j]
			F += b_sim + t_sim + e_sim
	return F / (Ns - k)



