"""
Module for writing to files.
"""

import os
import re
from typing import Union

import numpy as np

from matmdl.core.parser import uset
from matmdl.core.state import state


def write_params_to_file(param_values: list[float], param_names: list[str]) -> None:
	"""Appends last iteration params to file `out_progress.txt`."""

	opt_progress_header = ["time_ns"] + param_names
	out_fpath = os.path.join(uset.main_path, "out_progress.txt")

	add_header = not os.path.isfile(out_fpath)
	with open(out_fpath, "a+") as f:
		if add_header:
			header_padded = [opt_progress_header[0] + 12 * " "]
			for col_name in opt_progress_header[1:]:
				num_spaces = 8 + 6 - len(col_name)
				# 8 decimals, 6 other digits
				header_padded.append(col_name + num_spaces * " ")
			f.write(", ".join(header_padded) + "\n")
		line_string = ", ".join([f"{a:.8e}" for a in param_values]) + "\n"
		state.update_write()
		line_string = str(state.last_updated) + ", " + line_string
		f.write(line_string)


def combine_SS(zeros: bool, orientation: str) -> None:
	"""
	Reads npy stress-strain output and appends current results.

	Loads from ``temp_time_disp_force_{orientation}.csv`` and writes to
	``out_time_disp_force_{orientation}.npy``. Should only be called after all
	orientations have run, since ``zeros==True`` if any one fails.

	For parallel, needs to be called within a parallel.Checkout guard.

	Args:
	    zeros: True if the run failed and a sheet of zeros should be written
	        in place of real time-force-displacement data.
	    orientation: Orientation nickname to keep temporary output files separate.
	"""
	filename = os.path.join(uset.main_path, f"out_time_disp_force_{orientation}.npy")
	sheet = np.loadtxt(f"temp_time_disp_force_{orientation}.csv", delimiter=",", skiprows=1)
	if zeros:
		sheet = np.zeros((np.shape(sheet)))
	if os.path.isfile(filename):
		dat = np.load(filename)
		dat = np.dstack((dat, sheet))
	else:
		dat = sheet
	np.save(filename, dat)


def write_error_to_file(
	error_list: list[float], orient_list: list[str], combination_function
) -> None:
	"""
	Write error values separated by orientation, if applicable.

	Args:
	    error_list: List of floats indicated error values for each orientation
	        in ``orient_list``, with which this list shares an order.
	    orient_list: List of strings holding orientation nicknames.
	"""
	error_fpath = os.path.join(uset.main_path, "out_errors.txt")
	if not os.path.isfile(error_fpath):
		with open(error_fpath, "w+") as f:
			f.write(f"# errors for {orient_list} and combined error\n")

	with open(error_fpath, "a+") as f:
		f.write(
			",".join([f"{err:.8e}" for err in error_list + [combination_function(error_list)]])
			+ "\n"
		)


def write_input_params(
	fname: str,
	param_names: Union[list[str], str],
	param_values: Union[list[float], float],
	debug=False,
) -> None:
	"""
	Write parameter values to file with ``=`` as separator.

	Used for material and orientation input files.

	Args:
	    fname: Name of file in which to look for parameters.
	    param_names: List of strings (or single string) describing parameter names.
	        Shares order with ``param_values``.
	    param_values: List of parameter values (or single value) to be written.
	        Shares order with ``param_names``.

	Note:
	    Finds first match in file, so put derived parameters at end of file.
	"""
	match uset.format:
		case "huang":
			separator = " = "
		case "fepx":
			separator = " "
	if type(param_names) not in (list, tuple) and type(param_values) not in (
		list,
		tuple,
	):
		param_names = [param_names]
		param_values = [param_values]
	elif len(param_names) != len(param_values):
		raise IndexError("Length of names must match length of values.")

	with open(fname, "r") as f1:
		lines = f1.readlines()

	newlines = lines.copy()
	for param_name, param_value in zip(param_names, param_values):
		for i, line in enumerate(lines):
			match = re.search(r"\b" + param_name + r"[ |=]", line)
			if match:
				newlines[i] = (
					line[0 : match.start()] + param_name + separator + str(param_value) + "\n"
				)
				break  # go on to next param set

	with open("temp_" + fname, "w+") as f2:
		f2.writelines(newlines)

	if not debug:
		os.remove(fname)
		os.rename("temp_" + fname, fname)
