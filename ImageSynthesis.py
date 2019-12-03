import cv2
import numpy as np

def average_image(N, img = []):
	im = np.zeros((img[0].shape[0], img[0].shape[1], img[0].shape[2]), np.float64)
	for i in range(N):
		im += (img[i] / N)
	return im

def maximum_image(N, img = []):
	im = np.zeros((N, img[0].shape[0], img[0].shape[1], img[0].shape[2]), np.uint8)
	for i in range(N):
		im[i] = img[i]
	max_image = im.max(axis = 0)
	return max_image