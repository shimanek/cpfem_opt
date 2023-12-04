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
	if uset.main_path not in [os.getcwd(), "."]:
		print("Starting as a parallel instance")
		copy_files()
		# TODO: reload copied input.toml

"""
TODO: update output state with dates for each file modified:
 e.g., os.path.getmtime(fpath)
 if updated recently, reload optimizer

NOTE: single thread version works now,
- test with subfolders
- add above reloading
- test all together
- note some timing/performance stats

ERRORS:
- excess writing to npy files when out_progress hasn't been updated!

"""

def update_parallel(opt):
	""" state if dict of filename: linux seconds of last modification"""
	if uset.main_path in [os.getcwd(), "."]:
		return

	global output_state
	if output_state not in globals():
		output_state = _get_output_state()
	else:
		new_state = _get_output_state()
		state_diffs = [val1 != val2 for val1, val2 in (output_state, new_state)]
		if any(state_diffs):
			output_state = new_state
			# also update optimizer...
			# TODO: first column should be unique ID (time?) to facilitate diffs
			# for difflines in diff(internal prog, external prog):
			# do opt.tell(difflines[1:-1], difflines[-1])


def _get_output_state():
	output_state = {}
	outfiles = [f for f in os.path.listdir(uset.main_path) if f.startswith("out")]
	for fname in outfiles:
		fpath = os.path.join(uset.main_path, fname)
		output_state[f] = os.path.getmtime(fpath)
	return output_state


def copy_files():
	""" copy files from uset.main_path to runner dir"""
	#TODO: add experimental files for non-orientation case?
	
	# exact filenames
	flist = ["input.toml", uset.umat, uset.param_file, uset.jobname+".inp"]
	for orient in uset.orientations.keys():
		flist.append(uset.orientations[orient]["exp"])
		try:
			flist.append(uset.orientations[orient]["inp"])
		except KeyError:
			# orientation generated, no input file needed
			pass
	
	# only start of filenames
	fstarts = ["mesh", "mat"]
	for f in os.listdir(uset.main_path):
		for start in fstarts:
			if f.startswith(start):
				flist.append(f)

	# copy files to current directory
	for f in flist:
		copy(os.path.join(uset.main_path, f), os.getcwd())


class Checkout:
	"""checkout shared resource without write collisions"""
	def __init__(self, fname, local=False):
		self.fname = fname
		if local:
			self.fpath = os.path.join(os.getcwd(), fname)
		else:
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

	def __exit__(self, exc_type, exc_value, exc_tb):
		os.remove(self.fpath + ".lck")
