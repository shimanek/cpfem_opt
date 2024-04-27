"""
Module for dealing with the present optimization being one of many simultaneous instances.
This is presumed to be the case when the setting `main_path` has a value.
Everything here should be called within a Checkout guard.
"""

import os
import random
import re
import time
from shutil import copy

import numpy as np

from matmdl.core.parser import uset
from matmdl.core.state import state
from matmdl.core.utilities import msg, warn
from matmdl.engines import file_patterns


def check_parallel():
	"""
	Starts parallel initialization if needed.

	Note:
		This copies files from `uset.main_path` but does not reload the input file.
	"""
	if uset.main_path not in [os.getcwd(), "."]:
		msg("Starting as a parallel instance")
		copy_files(file_patterns)


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


def update_parallel():
	"""
	Update state if needed based on shared database timing information.

	Returns:
		params (list): parameter values (list) of unseen points to be updated
		errors (list): error values (scalar) of unseen points to be updated

	Note:
		Also updates `state.last_updated` timing information.
	"""
	if uset.main_path in [os.getcwd(), "."]:
		return ([], [])

	num_newlines = _get_num_newlines()
	if num_newlines < 1:
		return ([], [])

	# update state:
	num_lines = _get_totlines()
	start_line = num_lines - num_newlines + 1
	update_params = np.loadtxt(
		os.path.join(uset.main_path, "out_progress.txt"),
		delimiter=",",
		skiprows=start_line,
	)
	update_errors = np.loadtxt(
		os.path.join(uset.main_path, "out_errors.txt"),
		delimiter=",",
		skiprows=start_line,
	)

	# strict output database assertion:
	# assert_db_lengths_match()

	# quick assertion for params and errors only:
	has_multiple = len(np.shape(update_params)) == 2
	len_params = np.shape(update_params)[0] if has_multiple else 1
	len_errors = np.shape(update_errors)[0] if has_multiple else 1
	# ^ (if shape is 1D then there is only one entry)
	assert (
		len_params == len_errors
	), f"Error: mismatch in output database size! Found {len_params} params and {len_errors} errors"

	update_params_pass = []
	update_errors_pass = []
	if has_multiple:
		for i in range(np.shape(update_params)[0]):
			update_params_pass.append(list(update_params[i, 1:]))  # first value is time
			update_errors_pass.append(float(update_errors[i, -1]))  # last value is mean
	else:
		update_params_pass.append(list(update_params[1:]))
		update_errors_pass.append(float(update_errors[-1]))

	state.update_read()
	return update_params_pass, update_errors_pass


def assert_db_lengths_match():
	"""loads and checks lengths of all output files; used for debugging"""
	lengths = []
	for npyfile in [f for f in os.listdir(uset.main_path) if f.endswith("npy")]:
		dat = np.load(os.path.join(uset.main_path, npyfile))
		lengths.append(np.shape(dat)[2])
	for outfile in [
		f for f in os.listdir(uset.main_path) if f.startswith("out_") and f.endswith(".txt")
	]:
		dat = np.loadtxt(os.path.join(uset.main_path, outfile), delimiter=",", skiprows=1)
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


def copy_files(file_patterns):
	"""copy files from uset.main_path to runner dir"""

	# exact filenames
	flist = ["input.toml"]
	for orient in uset.orientations.keys():  # no need for ordering here
		flist.append(uset.orientations[orient]["exp"])
		try:
			flist.append(uset.orientations[orient]["inp"][0])
		except KeyError:
			# orientation generated, no input file needed
			pass

	# file list defined in engines:
	for f in os.listdir(uset.main_path):
		for pattern in file_patterns:
			if re.search(pattern, f):
				flist.append(f)

	# copy files to current directory
	for f in flist:
		copy(os.path.join(uset.main_path, f), os.getcwd())


class Checkout:
	"""Checkout shared resource without write collisions."""

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

		while True and time.time() - self.start < cutoff_seconds:
			lockfile_exists = os.path.isfile(self.fpath + ".lck")
			if lockfile_exists:
				try:
					with open(self.fpath + ".lck", "r") as f:
						source = f.read()
					msg(
						f"Waiting on Checkout for {time.time()-self.start:.3f} seconds from {source}"
					)
				except FileNotFoundError:
					msg(f"Waiting on Checkout for {time.time()-self.start:.3f} seconds")
				time.sleep(1)
			else:
				with open(self.fpath + ".lck", "a+") as f:
					f.write(f"{os.getcwd()}\n")
				self.time_unlocked = time.time()
				# check for collisions
				time.sleep(0.010)  # allow potential collision cases to catch up
				try:
					with open(self.fpath + ".lck", "r") as f:
						lines = f.readlines()
				except FileNotFoundError:
					lines = []
				if len(lines) != 1:
					warn("Warning: collision detected between processes:", RuntimeWarning)
					for line in lines:
						print(f"\t{line}", flush=True)
					warn("Reattempting to checkout resource", RuntimeWarning)
					try:
						os.remove(self.fpath + ".lck")
					except FileNotFoundError:
						pass  # only one process will successfully remove file
					time.sleep(4.0 * random.random())  # wait for a sec before restarting
					self.__enter__()  # try again

				msg(f"Unlocked after {time.time()-self.start:.3f} seconds")
				break
		if time.time() - self.start > cutoff_seconds:
			raise RuntimeError(
				f"Error: waited for resource {self.fname} for longer than {cutoff_seconds}s, exiting."
			)

	def __exit__(self, exc_type, exc_value, exc_tb):
		if False:  # debugging
			with open(self.fpath + ".lck", "r") as f:
				source = f.read()
			print(f"Exit: rm lock from: {source}", flush=True)
		os.remove(self.fpath + ".lck")
		msg(f"Exiting Checkout after {time.time()-self.time_unlocked:.3f} seconds.")

	def __call__(self, fn):
		"""
		Decorator to use if whole function needs resource checked out.
		"""

		def decorator():
			with self:
				return fn()

		return decorator
