#-------------------------------------------------------------------------------
# Name:         vector
# Purpose:      common functions for 1-D vectors
#
# Author:		ISet Group
#
# Created:		2016-06-03
#-------------------------------------------------------------------------------

import numpy as np

# return the color vector for specified number of classes
# input:	cnum, number of predefined classes
# output:   list of <rgb> for the classes
def color_map(cnum):
	delta = int(255*6/cnum)
	delta = np.min((delta, 255))

	cmap = np.zeros((cnum,3), np.uint8)
	base = 0

	for k in range(cnum):
		if k>0 and k%6==0:
			base += delta
		if k%3==0 or k%6==4:
			cmap[k,2] = base + delta
		else:
			cmap[k,2] = base
		if k%3==1 or k%6==5:
			cmap[k,1] = base + delta
		else:
			cmap[k,1] = base
		if k%3==2 or k%6==3:
			cmap[k,0] = base + delta
		else:
			cmap[k,0] = base

	return cmap


# compute statistics based on the classification of segments
# input:    L, the label matrix for the segments, start from 1
#           C, the list of corresponding classes, start from 1
#           cnum, number of classes
# output:   infos, the list of statistics, adding class 0 (unknown)
def compute_statistics(L, C, cnum = 0):
	if cnum <= 0:
		cnum = np.max(C)
	infos = []
	d = 1.0 * L.shape[0] * L.shape[1]
	mask = np.zeros(L.shape, np.uint16)
	for idx in range(len(C)):
		mask[L == idx+1] = C[idx]
	for k in range(cnum):
		s = np.count_nonzero(mask == k+1)
		infos.append(round(s*100/d, 3))
	infos.append(round(np.count_nonzero(mask == 0)*100/d, 3))

	return infos


# compute euclid distance between two vectors
# input v1, v2, two 1-dim vectors
# output: the euclid distance
def euclid_dist(v1, v2):
	d = np.asarray(v1) - np.asarray(v2)
	return np.sum(d*d)


# compute kullback-leibler distance between two distributions
# input v1, v2, two 1-dim vectors
# output: the kullback-leibler distance
def kullback_leibler_divergence(v1, v2):
	p = list(map(float, v1))
	q = list(map(float, v2))
	d = np.sum(np.abs(p))
	if d > 0:
		p /= d
	d = np.sum(np.abs(q))
	if d > 0:
		q /= d
	filt = np.logical_and(p != 0, q != 0)
	return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


# compute jensen-shannon distance between two distributions
# input v1, v2, two 1-dim vectors
# output: the jensen-shannon distance
def jensen_shannon_divergence(v1, v2):
	m = 0.5 * (np.asarray(v1) + np.asarray(v2))
	jsd = 0.5 * (kullback_leibler_divergence(v1, m) + kullback_leibler_divergence(v2, m))
	return jsd
