#-------------------------------------------------------------------------------
# Name:			unionFind
# Purpose:      implement the union-find algorithm
#               !!! unit id starts from 1 !!!
#
# Author:		ISet Group
#
# Created:		2015-12-24
#-------------------------------------------------------------------------------
import numpy as np

# object definition for the union-find
class UnionFinder(object):
	# ------ contructor ------------
	# num, number of elements
	def __init__(self, num):
		# ------- the fields ---------
		# start from 1, unit 0 is ignored
		self.flist = list(range(num+1))

		self.list = []
		for i in range(num+1):
			self.list.append([])
		for i in range(1, num+1):
			self.list[i].append(i)

	# connect two units into one component
	# input:	p, q, id of the two units
	# output:   none
	def merge(self, p, q):
		r1 = self.find(p)
		r2 = self.find(q)
		if r1 != r2:
			self.flist[r2] = r1
			n = len(self.list[r2])
			for i in range(n):
				self.list[r1].append(self.list[r2][0])
				self.list[r2].remove(self.list[r2][0])

	# find the root (of a component) id of a unit
	# input:	p, the unit id
	# output:   r, the root (representative) id
	def find(self, p):
		r = self.flist[p]
		if r != p:
			r = self.find(r)
		return r

	# get intital-final label pairs
	# input: sn, initial number of the new units
	# output: kv, key-value for initial-final labels
	def get_kv(self, sn=1):
		kv = {}
		curl = sn
		for s in range(1, len(self.flist)):
			root = self.find(s)
			if not root in kv:
				kv[root] = curl
				curl += 1
			if s != root:
				kv[s] = kv[root]
		return kv
