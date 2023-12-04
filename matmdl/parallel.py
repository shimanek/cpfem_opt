"""
Module for dealing with the present optimization being one of many simultaneous instances.
This is presumed to be the case when the setting `main_path` has a value.
"""
from contextlib import contextmanager
from matmdl.parser import uset
from shututil import copy
import os
import time


def check_parallel():
	"""parallel initialization if needed"""
	if uset.main_path is not os.getcwd():
		print("Starting as a parallel instance")
		copy_files()
		# TODO: reload copied input.toml


#TODO: update output state with dates for each file modified:
# e.g., os.path.getmtime(fpath)


def copy_files():
	""" copy files from uset.main_path to runner dir"""

	# get list of files to copy
	flist = ["input.toml", uset.umat, uset.param_file, uset.jobname+".inp"]
	for orient in uset.orientations.keys():
		try:
			flist.append(uset.orientations[orient]["inp"])
		except KeyError:
			# orientation generated, no input file needed
			pass

	# copy files to current directory
	for f in flist:
		copyfile(os.path.join(uset.main_path, f), os.getcwd())


@contextmanager
def checkout(fname):
	"""checkout shared resource without write collisions"""
	cutoff_seconds = 60

	start = time.time()
	fpath = os.path.join(uset.main_path, fname)

	while True and time.time() - start < cutoff_seconds:
		lockfile_exists = os.path.isfile(fpath + ".lck")
		if not lockfile_exists:
			try:
				open(fpath + ".lck", "w")
				f = open(fpath, "w+")
				yield f
			finally:
				f.close()
				os.remove(fpath + ".lck")
				break
		else:
			time.sleep(2)
