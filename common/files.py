#-------------------------------------------------------------------------------
# Name:         files
# Purpose:      the input/output functions
#
# Author:		ISet Group
#
# Created:		2015-12-03
#-------------------------------------------------------------------------------

import numpy as np
import csv
import string
import os

# get base name of a file
# input:    fname, dir name of the file
# output:   [basename, extention] of the file name
def get_file_names(fname):
	strs = os.path.basename(fname).split(".")
	prefix = strs[0]
	ext = strs[-1]
	return [prefix, ext]


# get the files containing the bounded and marked image
# input:    fname, dir name of the image file
# output:   [bname, mname], names of files containing the corresponding images
def get_fellow_names(fname):
	prefix = os.path.splitext(fname)[0]
	bname = prefix + '_bond.bmp'
	mname = prefix + '_mask.bmp'
	return [bname, mname]


# load a list of images/instances from a directory
# input:    root, directory containing the image/instance files
#           	 sub-directory as classes, image as instances
# output:   instances, a list of name pairs [fname, cname]
def read_instances(root):
	instances = []
	dirs = os.listdir(root)
	for d in dirs:
		dname = os.path.join(root, d)
		if not os.path.isdir(dname):
			continue
		files = os.listdir(dname)
		instances.extend([[os.path.join(dname, f), d] for f in files])

	return instances


# read in csv file as 2-dim int array
# input:    fname, file containing the labels for super pixels
# output:   mtx, 2-dim array for the labels
def read_csv_int(fname):
	mtx = []
	with open(fname) as infile:
		reader = csv.reader(infile)
		for line in reader:
			mtx.append([str2int(t) for t in line])
	infile.close()
	return np.array(mtx)


# read in csv file as the training data, features + labels
# input:    fname, file containing the training data
# output:   data, a 2-unit cell containing features + labels as arrays
#           	  !!! features and labels are lists !!!
def read_csv_data(fname):
	F = []
	CL = []
	with open(fname) as infile:
		reader = csv.reader(infile)
		for line in reader:
			F.append([str2float(t) for t in line[:-1]])
			CL.append(line[-1])
	infile.close()
	return (F, CL)


# write out training data (features + labels) to a csv file
# input:    data, a 2-unit cell containing features + labels
#           	  !!! features and labels are lists !!!
# 			fname, full name for the local file
# output: local csv file by @fname containing the training data
def write_csv_data(data, fname):
	outfile = file(fname, 'wb')
	writer = csv.writer(outfile)
	num = len(data[0])
	x = np.empty(num, list)
	for idx in range(num):
		x[idx] = []
		x[idx].extend(data[0][idx])
		x[idx].append(data[1][idx])
	writer.writerows(x)
	outfile.close()


# change a string value to int, default 0
# input:    s, string
# output:   v, an integer
def str2int(s):
	try:
		v = string.atoi(s)
		#v = int(s)
	except ValueError:
		v = 0
	return v


# change a string value to float, default 0
# input:    s, string
# output:   v, a float
def str2float(s):
	try:
		v = string.atof(s)
		#v = float(s)
	except ValueError:
		v = 0
	return v
