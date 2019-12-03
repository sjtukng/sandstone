import cv2
import csv
import ImageSynthesis
import numpy as np
import scipy.io as sio
import numpy as np
from images import Image

from common import unionFind
from segment import mslic, regions, clusters, dbscan
from scipy.stats import entropy
from skimage import segmentation
from measure import performance
from segment import features


def gt_generator(bond, fileName):
	cv2.imwrite(fileName, bond*255)

def mc_slic_test():
	im1 = cv2.imread("2.jpg", cv2.IMREAD_COLOR)
	im2 = cv2.imread("3.jpg", cv2.IMREAD_COLOR)
	im3 = cv2.imread("4.jpg", cv2.IMREAD_COLOR)
	im4 = cv2.imread("5.jpg", cv2.IMREAD_COLOR)
	im5 = cv2.imread("6.jpg", cv2.IMREAD_COLOR)
	im6 = cv2.imread("7.jpg", cv2.IMREAD_COLOR)
	im7 = cv2.imread("8.jpg", cv2.IMREAD_COLOR)
	max_im = ImageSynthesis.maximum_image(7, [im1, im2, im3, im4, im5, im6, im7])
	# Multi-channel SLIC
	img = [im1, im2, im3, im4, im5, im6, im7]
	L_mslic = mslic.mslic(img, 500, 5, 1, 5)
	L_mslic = regions.distinct_label(L_mslic)
##	L_mslic = clusters.merge_tiny_regions(im1, L_mslic, 180)

	contour = segmentation.mark_boundaries(max_im, L_mslic, (0, 0, 1))
	cv2.imshow("after mslic", contour)
	cv2.imwrite("after_mslic.jpg", contour * 255)
	cv2.waitKey(0)

def visual_experiment():

	im1 = cv2.imread("0degree.bmp", cv2.IMREAD_COLOR)
	im2 = cv2.imread("15degree.bmp", cv2.IMREAD_COLOR)
	im3 = cv2.imread("30degree.bmp", cv2.IMREAD_COLOR)
	im4 = cv2.imread("45degree.bmp", cv2.IMREAD_COLOR)

	max_im = ImageSynthesis.maximum_image(4, [im1, im2, im3, im4])

	# Efficient graph-based image segmentation, IJCV 2004
##	print "begin felzenszwalb"
##	scale = 200
##	sigma = 3
##	min_size = 100
##	L_fh = segmentation.felzenszwalb(max_im, scale, sigma, min_size)
##	contour_fh = segmentation.mark_boundaries(max_im, L_fh, (0, 0, 1))
##	cv2.imwrite("contour_fh.jpg", contour_fh * 255)
##	cv2.imshow("FH", contour_fh)
##	cv2.waitKey(0)
##
	# QuickShift method, ECCV 2008
##	print "begin quickshift"
##	ratio = 1.0
##	kernel_size = 10
##	max_dist = 5
##	return_tree = False
##	sigma = 3
##	convert2lab = True
##	random_seed = 42
##	L_qs = segmentation.quickshift(max_im, ratio, kernel_size, max_dist, return_tree, sigma, convert2lab, random_seed)
##	contour_qs = segmentation.mark_boundaries(max_im, L_qs, (0, 0, 1))
##	cv2.imwrite("contour_qs.jpg", contour_qs * 255)
##	cv2.imshow("QS", contour_qs)
##	cv2.waitKey(0)
##
	# SLIC superpixel method, TPAMI 2012
##	ns = 500
##	comp = 10
##	L_slic = segmentation.slic(max_im, n_segments=ns, compactness=comp, sigma=3, convert2lab=True)
##	L_slic = regions.distinct_label(L_slic)
##	L_slic = clusters.merge_tiny_regions(im1, L_slic, 500)
##	contour_slic = segmentation.mark_boundaries(max_im, L_slic, (0, 0, 1))
##	cv2.imwrite("slic.jpg", contour_slic * 255)
##	cv2.imshow("SLIC", contour_slic)
##	cv2.waitKey(0)

##	L_slic = clusters.merge_regions(im1, L_slic, iteration = 300)
##	contour3 = segmentation.mark_boundaries(max_im, L_slic, (0, 0, 1))
##	cv2.imshow("after regions merging", contour3)
##
##	cv2.waitKey(0)

 	# Seeds, IJCV 2015
##	seeds = cv2.ximgproc.createSuperpixelSEEDS(800, 600, 3, 500, 4, 2, 10)
##	seeds.iterate(max_im, 5)
##	L_seeds = seeds.getLabels()
##	contour_seeds = segmentation.mark_boundaries(max_im, L_seeds, (0, 0, 1))
##	cv2.imwrite("seeds.jpg", contour_seeds * 255)
##	cv2.imshow("Seeds", contour_seeds)
##	cv2.waitKey(0)

	# LSC, CVPR 2015
##	lsc = cv2.ximgproc.createSuperpixelLSC(max_im, 30, 0.075)
##	lsc.iterate(10)
##	L_lsc = lsc.getLabels()
##	L_lsc = regions.distinct_label(L_lsc)
##	L_lsc = clusters.merge_tiny_regions(im1, L_lsc, 500)
##	contour_lsc = segmentation.mark_boundaries(max_im, L_lsc, (0, 0, 1))
##	contour_lsc = mark_superpixel_no(contour_lsc, L_lsc)
##	cv2.imshow("LSC", contour_lsc)
##	cv2.imwrite("contour_lsc.jpg", contour_lsc * 255)
##	L_lsc = clusters.merge_regions(im1, L_lsc, iteration = 350)
##	contour3 = segmentation.mark_boundaries(max_im, L_lsc, (0, 0, 1))
##	cv2.imshow("after regions merging", contour3)
##	cv2.waitKey(0)

	# Multi-channel SLIC + Region merging, Our method
##	img = [im1, im2, im3, im4]
##	L_mslic = mslic.mslic(img, 500, 5, 1, 5)
##	L_mslic = regions.distinct_label(L_mslic)
##	L_mslic = clusters.merge_tiny_regions(im1, L_mslic, 200)
##	L_mslic = regions.distinct_label(L_mslic)
##
##	N = np.max(L_mslic)
##	print N
##	contour1 = segmentation.mark_boundaries(max_im, L_mslic, (0, 0, 1))
##	contour1 = mark_superpixel_no(contour1, L_mslic)
##	cv2.imshow("after mslic", contour1)
##	cv2.imwrite("after mslic.jpg", contour1 * 255)
##	L_mslic2 = mslic.merge_regions(img, L_mslic, iteration = 150)
##	L_mslic3 = clusters.merge_regions(img, L_mslic, iteration = 600)
##	contour = segmentation.mark_boundaries(max_im, L_mslic2, (0, 0, 1))
##	contour3 = segmentation.mark_boundaries(max_im, L_mslic3, (0, 0, 1))
##	cv2.imwrite("after region merging.jpg", contour3 * 255)
##	cv2.imshow("after region merging", contour3)
##	cv2.imshow("contour3", contour2)
##	cv2.waitKey(0)


def precision_recall():

	im1 = cv2.imread("data/2-0.jpg", cv2.IMREAD_COLOR)
	im2 = cv2.imread("data/2-15.jpg", cv2.IMREAD_COLOR)
	im3 = cv2.imread("data/2-30.jpg", cv2.IMREAD_COLOR)
	im4 = cv2.imread("data/2-45.jpg", cv2.IMREAD_COLOR)
	gt = cv2.imread("gt2.jpg", 0)

	# small image for test only
##	im1 = cv2.imread("0.bmp", cv2.IMREAD_COLOR)
##	im2 = cv2.imread("15.bmp", cv2.IMREAD_COLOR)
##	im3 = cv2.imread("30.bmp", cv2.IMREAD_COLOR)
##	im4 = cv2.imread("45.bmp", cv2.IMREAD_COLOR)
##	gt = cv2.imread("gt2.jpg", 0)

	thres = 3       # threshold of distance between edge pixel and g.t. edge pixel
	# Multi-channel SLIC, our method
	# print ("begin multi-channel slic")
	# img = [im1, im2, im3, im4]
	# L = mslic.mslic(img, 300, 5, 1, 5)
	# L = regions.distinct_label(L)
	# print ("after multi-channel slic, there are " + str(np.max(L)) + " superpixels")
	# L = clusters.merge_tiny_regions(im1, L, 200)
	# print ("after merging tiny regions, there are " + str(np.max(L)) + " superpixels left")
	# bond_L = segmentation.find_boundaries(L)
	# cv2.imwrite("after mslic.jpg", bond_L*255)

	# print (performance.boundary_recall(gt, bond_L, thres))
	# print (performance.precision(gt, bond_L, thres))

	# L = clusters.merge_regions(img, L, gt, iteration = 100)

	# cv2.imwrite("it=309.jpg", contour * 255)
	# print (performance.boundary_recall(gt, bond_L, thres))
	# print (performance.precision(gt, bond_L, thres))



	# QuickShift method, ECCV 2008
##	print "begin quickshift"
##	ratio = 1.0
##	kernel_size = 10
##	max_dist = 5
##	return_tree = False
##	sigma = 3
##	convert2lab = True
##	random_seed = 42
##	L_qs = segmentation.quickshift(max_im, ratio, kernel_size, max_dist, return_tree, sigma, convert2lab, random_seed)
##	print "after Quickshift, there are " + str(np.max(L_qs)) + " superpixels"
##	L = clusters.merge_tiny_regions(im1, L_qs, 200)
##	print "after merging tiny regions, there are " + str(np.max(L)) + " superpixels left"
##	bond_qs = segmentation.find_boundaries(L)
##	print performance.boundary_recall(gt, bond_qs, thres)
##	print performance.precision(gt, bond_qs, thres)
##	img = [im1, im2, im3, im4]
##	L = clusters.merge_regions(img, L, gt, iteration = 300)
##	bond_L = segmentation.find_boundaries(L, mode='thick')
##	print performance.boundary_recall(gt, bond_L, thres)
##	print performance.precision(gt, bond_L, thres)


	# Efficient graph-based image segmentation, IJCV 2004
	# print "begin Graph-based segmentation"
	# scale = 80
	# sigma = 2
	# min_size = 5
	# L_fh = segmentation.felzenszwalb(max_im, scale, sigma, min_size)
	# L = regions.distinct_label(L_fh)
	# print "after Graph-based segmentation, there are " + str(np.max(L)) + " superpixels"
	# contour = segmentation.mark_boundaries(max_im, L, (0, 0, 1))
	# cv2.imwrite("contour_fh.jpg", contour*255)
	# cv2.waitKey(0)

##	L = clusters.merge_tiny_regions(im1, L, 200)
##	print "after merging tiny regions, there are " + str(np.max(L)) + " superpixels left"
##	bond_fh = segmentation.find_boundaries(L)
##	print performance.boundary_recall(gt, bond_fh, thres)
##	print performance.precision(gt, bond_fh, thres)
##	img = [im1, im2, im3, im4]
##	L = clusters.merge_regions(img, L, gt, iteration = 200)
##	bond_L = segmentation.find_boundaries(L, mode='thick')
##	print performance.boundary_recall(gt, bond_L, thres)
##	print performance.precision(gt, bond_L, thres)


	# SLIC superpixel method, TPAMI 2012
	print ("begin slic")
	im = Image("data/2-0.jpg")
	ns = 300
	comp = 20
	L_slic = segmentation.slic(im1, n_segments=ns, compactness=comp, sigma=3, convert2lab=True)
	L = regions.distinct_label(L_slic)
	L = clusters.merge_tiny_regions(im1, L, 500)
	print ("after merging tiny regions, there are " + str(np.max(L)) + " superpixels left")
	contour = segmentation.mark_boundaries(im1, L, (0, 0, 1))
	# cv2.imshow("hehe", contour)
	# cv2.waitKey(0)


	print("start dbscan..............")
	L = dbscan.merge_region(im, L, [14, 1])
	print ("after dbscan, there are " + str(np.max(L)) + " superpixels left")
	contour = segmentation.mark_boundaries(im1, L, (0, 0, 1))
	cv2.imshow("hehe", contour)
	cv2.imwrite("feldspar_dbscan.jpg", contour * 255)
	cv2.waitKey(0)



#	bond_slic = segmentation.find_boundaries(L)
##	print performance.boundary_recall(gt, bond_slic, thres)
##	print performance.precision(gt, bond_slic, thres)
##	img = [im1, im2, im3, im4]
##	L = clusters.merge_regions(img, L, gt, iteration = 600)
##	bond_L = segmentation.find_boundaries(L, mode='thick')
##	print performance.boundary_recall(gt, bond_L, thres)
##	print performance.precision(gt, bond_L, thres)

	# Seeds, IJCV 2015
##	seeds = cv2.ximgproc.createSuperpixelSEEDS(800, 600, 3, 300, 4, 2, 20)
##	seeds.iterate(max_im, 100)
##	L_seeds = seeds.getLabels()
##	L = regions.distinct_label(L_seeds)
##	print "after SEEDS, there are " + str(np.max(L)) + " superpixels"
##	L = clusters.merge_tiny_regions(im1, L, 200)
##	print "after merging tiny regions, there are " + str(np.max(L)) + " superpixels left"
##	bond_seeds = segmentation.find_boundaries(L)
##	print performance.boundary_recall(gt, bond_seeds, thres)
##	print performance.precision(gt, bond_seeds, thres)
##	img = [im1, im2, im3, im4]
##	L = clusters.merge_regions(img, L, gt, iteration = 200)
##	bond_L = segmentation.find_boundaries(L, mode='thick')
##	print performance.boundary_recall(gt, bond_L, thres)
##	print performance.precision(gt, bond_L, thres)

	# LSC, CVPR 2015
##	lsc = cv2.ximgproc.createSuperpixelLSC(max_im, 30, 0.075)
##	lsc.iterate(20)
##	L_lsc = lsc.getLabels()
##	L = regions.distinct_label(L_lsc)
##	print "after LSC, there are " + str(np.max(L)) + " superpixels"
##
##	L = clusters.merge_tiny_regions(max_im, L, 200)
##	print "after merging tiny regions, there are " + str(np.max(L)) + " superpixels left"
##	contour = segmentation.mark_boundaries(max_im, L, (0, 0, 1))
##	cv2.imshow("hehe", contour)
##	cv2.imwrite("contour_lsc.jpg", contour * 255)
##	cv2.waitKey(0)
##	bond_lsc = segmentation.find_boundaries(L)
##	print performance.boundary_recall(gt, bond_lsc, thres)
##	print performance.precision(gt, bond_lsc, thres)
##	img = [im1, im2, im3, im4]
##	L = clusters.merge_regions(img, L, gt, iteration = 500)
##	bond_L = segmentation.find_boundaries(L, mode='thick')
##	contour = segmentation.mark_boundaries(max_im, L, (0, 0, 1))
##	print performance.boundary_recall(gt, bond_L, thres)
##	print performance.precision(gt, bond_L, thres)

	# TurboPixel
##	load_data = sio.loadmat('tp_boundary.mat')
##	bond_tp = load_data['boundary']
##	print performance.boundary_recall(gt, bond_tp, thres)
##	print performance.precision(gt, bond_tp, thres)

	# ERS
##	load_data = sio.loadmat('label_ers.mat')
##	L_ers = load_data['labels']
##	L = regions.distinct_label(L_ers)
##	print "after ERS, there are " + str(np.max(L)) + " superpixels"
##	L = clusters.merge_tiny_regions(im1, L, 200)
##	print "after merging tiny regions, there are " + str(np.max(L)) + " superpixels left"
##	bond_ers = segmentation.find_boundaries(L)
##	print performance.boundary_recall(gt, bond_ers, thres)
##	print performance.precision(gt, bond_ers, thres)
##	img = [im1, im2, im3, im4]
##	L = clusters.merge_regions(img, L, gt, iteration = 200)
##	bond_L = segmentation.find_boundaries(L, mode='thick')
##	print performance.boundary_recall(gt, bond_L, thres)
##	print performance.precision(gt, bond_L, thres)

def mark_superpixel_no(img, L):
	# write no of each superpixel in the image
	font = cv2.FONT_HERSHEY_SIMPLEX
	num = np.max(L)
	for idx in range(1, num + 1):
		mask = np.where(L == idx)
		x = np.mean(mask[0])
		y = np.mean(mask[1])
		cv2.putText(img, str(idx), (int(y), int(x)), font, 0.35, (255,0,0), 1)
	return img

def paramenter_comparison():

	im1 = cv2.imread("0degree.bmp", cv2.IMREAD_COLOR)
	im2 = cv2.imread("15degree.bmp", cv2.IMREAD_COLOR)
	im3 = cv2.imread("30degree.bmp", cv2.IMREAD_COLOR)
	im4 = cv2.imread("45degree.bmp", cv2.IMREAD_COLOR)
	gt = cv2.imread("gt.jpg", 0)
	max_im = ImageSynthesis.maximum_image(4, [im1, im2, im3, im4])
	thres = 3       # threshold of distance between edge pixel and g.t. edge pixel

	# Multi-channel SLIC, our method
	print ("begin multi-channel slic")
	img = [im1, im2, im3, im4]

##	K = [200, 250, 300, 350, 400, 450, 500, 550, 600]
##	m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

	K = [300]
	m = [5]

##	outfile = file('parameter analysis.csv', 'wb')
##	writer = csv.writer(outfile)

	for i in range(len(K)):
		for j in range(len(m)):
			L = mslic.mslic(img, K[i], m[j], 1, 5)
			L = regions.distinct_label(L)
			print ("after multi-channel slic, there are " + str(np.max(L)) + " superpixels")
			L = clusters.merge_tiny_regions(im1, L, 200)
			print ("after merging tiny regions, there are " + str(np.max(L)) + " superpixels left")
			bond_L = segmentation.find_boundaries(L)

			print ("K=" + str(K[i]) + " m=" + str(m[j]))
			recall = performance.boundary_recall(gt, bond_L, thres)
			precision = performance.precision(gt, bond_L, thres)
			print (str(recall))
			print (str(precision))

##			row = [K[i], m[j], recall, precision]
##			writer.writerow(row)
##	outfile.close()







if __name__ == '__main__':
##	visual_experiment()
	# mc_slic_test()
	precision_recall()
##	paramenter_comparison()

##	gt = cv2.imread("groundtruth2.jpg", 0)
##	gt[gt<200] = 0
##	gt[gt>=200] = 1
##	cv2.imwrite("gt2.jpg",gt)



