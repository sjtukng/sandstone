#-------------------------------------------------------------------------------
# Name:			option
# Purpose:      a class containing all the options
#
# Author:		ISet Group
#
# Created:		2016-06-03
#-------------------------------------------------------------------------------

import threading, os
from common import vectors

# Function: get the application base path
# Input: None
# Output: the base application path
def base_path(debug):
	if debug:
		dName = os.path.join('D:\MyTools', 'ContentSegment')
	else:
		dName = os.path.join(os.path.dirname(__file__), os.pardir)
	return os.path.abspath(dName)


# object definition for the Option
class Option(object):
	DEBUG = True
	cNum = 3
	classes = ['quartz', 'feldspar', 'detritus']  # vector of class names
	cmap = vectors.color_map(cNum)			# color vector, arranged in [b-g-r]
	baseDir = base_path(DEBUG)
	dataDir = os.path.join(baseDir, 'data')
	imgDir = os.path.join(dataDir, 'samples')
	grainDir = os.path.join(dataDir, 'grains')
	iconDir = os.path.join(baseDir, 'icon')
	splash = os.path.join(iconDir, 'logo.jpg')
	trainSet = 'sandgrains.csv'
	# the main frame
	mainWin = None
	event = threading.Event()
	# directly referenced by the class
	target = 'rock1.jpg'
	# general parameters
	clfName = 'RandomForest'
	algorithm = 'slic'
	tsize = 50
	mwin = 3
	iterate = 10
	# for slic
	sNseg = 300
	sCompact = 15
	sSigma = 3
	# for quickshift
	qRatio = 1.0
	qKernel = 5
	qDist = 10
	qSeed = 42
	# for k-means
	kCore = 7
	kAttempts = 3
	# for iterative grabcut
	gDepth = 3
	# for super-pixel LSC
	lRatio = 0.075
	# for super-pixel Seeds
	sLevels = 7
	sPrior = 2
	sBins = 10
	sDouble = True
	# for region growth
	rDist = 3
	rEdge = 0
	rAlpha = 9
	rRev = True
	# for lbp transformation
	radius = 7
	nPoints = 8 * radius
	lpbMeth = 'uniform'
	# for mslic
	mCompact = 5
	mIter = 5
	mSize = 300
	# for coarse-fine merging
	cThB = 0.1
	cThF = 0.00001
	cAlpha = 0.001
	cCNum = 5
	# brightness, hist, texture
	cFeats = [False, True, False, False]
	# for dbscan
	dbTresh = 15
	dbText = 0.5
	# for classification
	thresh = 0.5    # threshold for classifications

	# compute work load
	# input:	desc, description of the work for waiting slidebar
	# output:   wl, number of works for the slidebar
	@classmethod
	def work_load(cls, desc):
		wl = 20
		if desc == 'slic':
			wl = 10
		elif desc == 'quickshift':
			wl = 15
		elif desc == 'k-means':
			wl = 8
		elif desc == 'regrowth':
			wl = 25
		elif desc == 'grabcut':
			wl = 20
		elif desc == 'lsc':
			wl = 10
		elif desc == 'seeds':
			wl = 8
		elif desc == 'cofim':
			wl = 36
		elif desc == 'classify':
			wl = 15
		elif desc == 'training':
			wl = 5
		elif desc == 'openimage':
			wl = 3
		elif desc == 'bench':
			wl = 36
		else:
			print('!!Warning: no work load defined for', desc)
		return wl

	# ------ contructor ------------
	def __init__(self):
		# ------- the fields ---------
		pass
