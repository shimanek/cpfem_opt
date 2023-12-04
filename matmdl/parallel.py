"""
Module for dealing with the present optimization being one of many simultaneous instances.
This is presumed to be the case when the setting `main_path` has a value.
"""
from contextlib import contextmanager
from matmdl.parser import uset
from shutil import copy
import os
import time


def check_parallel():
	"""parallel initialization if needed"""
	print("dbg:par:", uset.main_path)
	if uset.main_path not in [os.getcwd(), "."]:
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
		copy(os.path.join(uset.main_path, f), os.getcwd())


class Checkout:
	"""checkout shared resource without write collisions"""
	def __init__(self, fname):
		self.fname = fname
		self.fpath = os.path.join(uset.main_path, fname)

	def __enter__(self):
		cutoff_seconds = 60
		start = time.time()

		while True and time.time() - start < cutoff_seconds:
			lockfile_exists = os.path.isfile(self.fpath + ".lck")
			if lockfile_exists:
				time.sleep(2)
			else:
				open(self.fpath + ".lck", "w")
				break
		if time.time() - start > cutoff_seconds:
			raise RuntimeError(f"Error: waited for resource {self.fname} for longer than {cutoff_seconds}, exiting.")

	def __exit__(self):
		os.remove(self.fpath + ".lck")
