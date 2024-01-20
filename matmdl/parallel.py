"""
Module for dealing with the present optimization being one of many simultaneous instances.
This is presumed to be the case when the setting `main_path` has a value.
Everything here should be called within a Checkout guard.
"""
from matmdl.parser import uset
from matmdl.state import state
import numpy as np
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

 TODO: put directory name on lockfile? for debug purposes. eg: `out_sub_01.lck`

NOTE: single thread version works now,
- test with subfolders
- add above reloading
- test all together
- note some timing/performance stats

ERRORS:
- excess writing to npy files when out_progress hasn't been updated!

"""

def _get_num_newlines():
	"""Check for updates; needs to be within Checkout guard."""
	num_newlines = 0
	fname = os.path.join(uset.main_path, "out_progress.txt")
	try:
		times = np.loadtxt(fname, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
	except FileNotFoundError:
		return 0

	if np.shape(times) == ():
		return 0

	for time in times:
		if time > state.last_updated:
			num_newlines += 1

	return num_newlines


def _get_totlines():
	"""Excluding header!"""
	totlines = -1
	with open(os.path.join(uset.main_path, "out_progress.txt"), "r") as f:
		for line in f:
			totlines += 1
	return totlines


def update_parallel(opt):
	""" state if dict of filename: linux seconds of last modification"""
	if uset.main_path in [os.getcwd(), "."]:
		return

	num_newlines = _get_num_newlines()
	if num_newlines < 1:
		return

	# update state:
	num_lines = _get_totlines()
	start_line = num_lines - num_newlines + 1
	update_params = np.loadtxt(os.path.join(uset.main_path, "out_progress.txt"), delimiter=',', skiprows=start_line)
	update_errors = np.loadtxt(os.path.join(uset.main_path, "out_errors.txt"), delimiter=',', skiprows=start_line)

	# strict output database assertion:
	# assert_db_lengths_match()
	# quick assertion for params and errors only:
	assert np.shape(update_params)[0] == np.shape(update_errors)[0], \
		f"Error: mismatch in output database size! Found {np.shape(update_params)[0]} params and {np.shape(update_errors)[0]} errors"

	update_params_pass = []
	update_errors_pass = []
	for i in range(np.shape(update_params)[0]):
		update_params_pass.append(list(update_params[i,1:]))  # first value is time
		update_errors_pass.append(float(update_errors[i,-1]))  # last value is mean

	opt.tell(update_params_pass, update_errors_pass)
	state.update_read()


def assert_db_lengths_match():
	"""loads and checks lengths of all output files; used for debugging"""
	lengths = []
	for npyfile in [f for f in os.listdir(uset.main_path) if f.endswith("npy")]:
		dat = np.load(os.path.join(uset.main_path, npyfile))
		lengths.append(np.shape(dat)[2])
	for outfile in [f for f in os.listdir(uset.main_path) if f.startswith("out_") and f.endswith(".txt")]:
		dat = np.loadtxt(os.path.join(uset.main_path, outfile), delimiter=',', skiprows=1)
		lengths.append(np.shape(dat)[0])

	if len(set(lengths)) > 1:
		error_time = time.time_ns()
		with open(os.path.join(uset.main_path, "out_progress.txt"), "a+") as f:
			f.write(f"{error_time}, ERROR from {os.getcwd()}")
		raise RuntimeError(f"mismatch in DB lengths at time: {error_time}")


def _get_output_state():
	output_state = {}
	outfiles = [f for f in os.path.listdir(uset.main_path) if f.startswith("out")]
	for fname in outfiles:
		fpath = os.path.join(uset.main_path, fname)
		output_state[fname] = os.path.getmtime(fpath)
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
		self.start = time.time()
		self.fname = fname
		if local:
			self.fpath = os.path.join(os.getcwd(), fname)
		else:
			self.source = os.getcwd()
			self.fpath = os.path.join(uset.main_path, fname)

	def __enter__(self):
		cutoff_seconds = 420
		start = time.time()

		while True and time.time() - start < cutoff_seconds:
			lockfile_exists = os.path.isfile(self.fpath + ".lck")
			if lockfile_exists:
				print(f"Waiting on Checkout for {time.time()-self.start} seconds.")
				time.sleep(2)
			else:
				open(self.fpath + ".lck", "w")
				# TODO try writing os.getcwd() to file
				break
		if time.time() - start > cutoff_seconds:
			raise RuntimeError(f"Error: waited for resource {self.fname} for longer than {cutoff_seconds}s, exiting.")

	def __exit__(self, exc_type, exc_value, exc_tb):
		os.remove(self.fpath + ".lck")
		print(f"Exiting Checkout after {time.time()-self.start} seconds.")

	def decorate(fname, local=True):
		"""
		Decorator to use if whole function needs resource checked out.

		TODO: this results in two calls to __exit__() but seems to be functional
		"""
		def _decorate(fn):
			def wrapper(fname, local=local):
				with Checkout(fname, local=local):
					return fn
			return wrapper(fname, local=local)
		return _decorate
