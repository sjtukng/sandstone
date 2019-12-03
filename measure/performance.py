#-------------------------------------------------------------------------------
# Name:        performance
# Purpose:     evaluate algorithm performance and visual output
#
# Author:      ISet Group
#
# Created:     30/08/2016
#-------------------------------------------------------------------------------

import numpy as np
import csv

#-------------------------------------------------------------------------------
# Function: compute boundary recall of a segmentation method
# Input:    gt, ground truth boundary map, 1 -- boundary, 0 -- non-boundary
#           bond, the boundary map given by the algorithm, 1 -- boundary, 0 -- non-boundary
#           thres, threshold of distance between edge pixel and g.t. edge pixel
#               default setting is 2
# Output:   br, boundary recall rate
#-------------------------------------------------------------------------------
def boundary_recall(gt, bond, thres = 2):

	width = gt.shape[1]
	height = gt.shape[0]

	boundary_idx = np.where(gt == 1)

	num_boundary_pixels = len(boundary_idx[0])
	num_recall = 0

	for i in range(num_boundary_pixels):
		x = boundary_idx[0][i]
		y = boundary_idx[1][i]
		neighbor = []
		neighbor.append([x, y])
		for i in range(1, thres + 1):
			x1 = x - i
			x2 = x + i
			y1 = y - i
			y2 = y + i
			if x1 >= 0:
				neighbor.append([x1, y])
			if x2 < height:
				neighbor.append([x2, y])
			if y1 >= 0:
				neighbor.append([x, y1])
			if y2 < width:
				neighbor.append([x, y2])

		for i in range(len(neighbor)):
			if bond[neighbor[i][0], neighbor[i][1]] > 0:
				num_recall += 1
				break

	return float(num_recall) / num_boundary_pixels


#-------------------------------------------------------------------------------
# Function: compute precision of a segmentation method
# Input:    gt, ground truth boundary map, 1 -- boundary, 0 -- non-boundary
#           bond, the boundary map given by the algorithm, 1 -- boundary, 0 -- non-boundary
#           thres, threshold of distance between edge pixel and g.t. edge pixel
#				default setting is 2
# Output:   pr, precision rate
#-------------------------------------------------------------------------------
def precision(gt, bond, thres = 2):
	width = gt.shape[1]
	height = gt.shape[0]

	predicted_idx = np.where(bond == 1)

	num_predicted_pixels = len(predicted_idx[0])
	num_true_positive = 0

	for i in range(num_predicted_pixels):
		x = predicted_idx[0][i]
		y = predicted_idx[1][i]
		neighbor = []
		neighbor.append([x, y])
		for i in range(1, thres + 1):
			x1 = x - i
			x2 = x + i
			y1 = y - i
			y2 = y + i
			if x1 >= 0:
				neighbor.append([x1, y])
			if x2 < height:
				neighbor.append([x2, y])
			if y1 >= 0:
				neighbor.append([x, y1])
			if y2 < width:
				neighbor.append([x, y2])

		for i in range(len(neighbor)):
			if gt[neighbor[i][0], neighbor[i][1]] > 0:
				num_true_positive += 1
				break
	return float(num_true_positive) / num_predicted_pixels



# write out precision/recall data to a csv file
# input:    data, a 2-unit cell containing precision + recall
#           	  !!! precision and recall are lists !!!
# 			fname, full name for the local file
# output: local csv file by @fname containing the precision/recall data
def write_pr_curve(data, fname):
	outfile = file(fname, 'wb')
	writer = csv.writer(outfile)
	num = len(data[0])
	x = np.empty(num, list)
	for idx in range(num):
		x[idx] = []
		x[idx].append(data[0][idx])
		x[idx].append(data[1][idx])
	writer.writerows(x)
	outfile.close()


