"""
Recalculates error values based on saved stress-strain data.

Reloads *.npy files to recalculation of individual error values.
Moves current out_errors.txt to dated filename before rewriting.
Uses current error settings
"""

import datetime
import os

import numpy as np

from matmdl.core import writer as writer
from matmdl.core.experimental import ExpData
from matmdl.core.parser import uset
from matmdl.core.utilities import warn
from matmdl.objectives import calc_error, combine_error


def recalculate():
	# change folder to do in place:
	with uset.unlock():
		uset.main_path = os.getcwd()

	# mv error file:
	try:
		date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		os.rename("out_errors.txt", f"out_errors_{date_string}.txt")
	except FileNotFoundError:
		warn("No previous error file found, continuing")

	# load previous data:
	exp_data = ExpData(uset.orientations)
	orients = sorted(list(uset.orientations.keys()))
	# ^ InOpt uses same sorting... not essential but nice to match
	data = {}
	lengths = []
	for orient in orients:
		data[orient] = np.load(f"out_time_disp_force_{orient}.npy", allow_pickle=False)
		lengths.append(np.shape(data[orient])[-1])

	# check that all input data have the same length:
	length = lengths[0]
	length_diffs = np.asarray([length - lengths[0] for length in lengths], dtype=bool)
	if any(length_diffs):
		raise ValueError("Found data files of different lengths, stopping.")

	# loop thru stress-strains
	for i in range(0, length):
		# do each comparison, add to new data file
		errors = []
		for orient in orients:
			errors.append(
				calc_error(exp_data.data[orient]["raw"], orient, sim_data=data[orient][1:, 1:, i])
			)
		writer.write_error_to_file(errors, orients, combine_error)


if __name__ == "__main__":
	recalculate()
