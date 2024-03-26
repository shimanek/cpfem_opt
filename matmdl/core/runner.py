"""
Helper module for some abstracted commands used in [run][matmdl.run].
"""

import os
import shutil
import subprocess
import sys
from typing import Union

import numpy as np

from matmdl import engines as engine
from matmdl.core import optimizer as optimizer
from matmdl.core import writer as writer
from matmdl.core.crystalPlasticity import get_orient_info
from matmdl.core.experimental import ExpData
from matmdl.core.parser import uset
from matmdl.core.state import state


def get_first(opt, in_opt, exp_data) -> None:
	"""
	Run one simulation so its output dimensions can later inform the shape of output data.
	"""
	# test with strain of 0.2%
	engine.write_strain(uset.length * 0.002, f"{uset.jobname}.inp")
	with state.TimeRun()():
		engine.run()
	if not engine.has_completed():
		refine_run()
	engine.extract("initial")
	# reset to first max_strain; if multiple samples, will be overwritten anyway
	first_sample = list(exp_data.data.keys())[0]
	engine.write_strain(exp_data.data[first_sample]["max_strain"], f"{uset.jobname}.inp")


def check_single():
	"""rough copy of run/single_loop that does not use an optimizer object"""
	if not uset.do_single:
		return
	print("DBG: starting single run!")

	# load options:
	in_opt = optimizer.InOpt(uset.orientations, uset.params)
	next_params = []
	exp_data = ExpData(uset.orientations)  # noqa: F841
	# above line to make main input files with correct strain magnitude

	# ck that there are no ranges in input
	for param_name, param_value in uset.params.items():
		if type(param_value) in [list, tuple]:
			raise TypeError(
				f"Expected prescribed parameters for single run; found parameter bounds for {param_name}"
			)

	engine.prepare()
	for orient in in_opt.orients:
		print(f"DBG: starting orient {orient}")
		if in_opt.has_orient_opt[orient]:
			orient_components = get_orient_info(next_params, orient, in_opt)
			writer.write_input_params(
				"mat_orient.inp",
				orient_components["names"],
				orient_components["values"],
			)
			shutil.copy("mat_orient.inp", f"mat_orient_{orient}.inp")
		else:
			try:
				shutil.copy(uset.orientations[orient]["inp"], f"mat_orient_{orient}.inp")
			except shutil.SameFileError:
				pass
		shutil.copy(f"mat_orient_{orient}.inp", "mat_orient.inp")
		shutil.copy(f"{uset.jobname}_{orient}.inp", f"{uset.jobname}.inp")

		engine.run()
		if not engine.has_completed():
			print(f"DBG: refining orient {orient}")
			refine_run()
		if not engine.has_completed():
			print(f"DBG: not complete with {orient}, exiting...")
			sys.exit(1)
		else:
			output_fname = f"temp_time_disp_force_{orient}.csv"
			if os.path.isfile(output_fname):
				os.remove(output_fname)
			engine.extract(orient)  # extract data to temp_time_disp_force.csv
			if np.sum(np.loadtxt(output_fname, delimiter=",", skiprows=1)[:, 1:2]) == 0:
				print(f"Warning: incomplete run for {orient}, continuing...")
				return
	print("DBG: exiting single run!")
	sys.exit(0)


def remove_out_files():
	"""Delete files from previous optimization runs if not reloading results."""
	if not uset.do_load_previous:
		out_files = [
			f
			for f in os.listdir(os.getcwd())
			if (f.startswith("out_") or f.startswith("res_") or f.startswith("temp_"))
		]
		if len(out_files) > 0:
			for f in out_files:
				os.remove(f)
	job_files = [
		f
		for f in os.listdir(os.getcwd())
		if (f.startswith(uset.jobname)) and not (f.endswith(".inp"))
	]
	for f in job_files:
		if os.path.isdir(f):
			os.rmdir(f)
		else:
			os.remove(f)


def refine_run(ct: int = 0):
	"""
	Restart simulation with smaller maximum increment size.

	Cut max increment size by ``factor`` (hardcoded), possibly multiple
	times up to ``uset.recursion_depth`` or until Abaqus finished successfully.
	After eventual success or failure, rewrites original input file so that the
	next run starts with the initial, large maximum increment.
	Recursive calls tracked through ``ct`` parameter.

	Args:
	    ct: Number of times this function has already been called. Starts
	        at 0 and can go up to ``uset.recursion_depth``.
	"""
	if uset.format == "fepx":
		# TODO should separate out all engine-specific calls
		# and raise NotImplemented errors from there if applicable
		return
	factor = 5.0
	ct += 1
	# remove old lock file from previous unfinished simulation
	subprocess.run("rm *.lck", shell=True)
	filename = uset.jobname + ".inp"
	tempfile = "temp_input.txt"
	with open(filename, "r") as f:
		lines = f.readlines()

	# exit strategy:
	if ct == 1:  # need to save original parameters outside of this recursive function
		with open(tempfile, "w") as f:
			f.writelines(lines)

	def write_original(filename):
		with open(tempfile, "r") as f:
			lines = f.readlines()
		with open(filename, "w") as f:
			f.writelines(lines)

	# find line after step line:
	step_line_ind = [i for i, line in enumerate(lines) if line.lower().startswith("*static")][0] + 1
	step_line = [number.strip() for number in lines[step_line_ind].strip().split(",")]
	original_increment = float(step_line[-1])

	# use original / factor:
	new_step_line = step_line[:-1] + ["%.4E" % (original_increment / factor)]
	new_step_line_str = str(new_step_line[0])
	for i in range(1, len(new_step_line)):
		new_step_line_str = new_step_line_str + ", "
		new_step_line_str = new_step_line_str + str(new_step_line[i])
	new_step_line_str = new_step_line_str + "\n"
	with open(filename, "w") as f:
		f.writelines(lines[:step_line_ind])
		f.writelines(new_step_line_str)
		f.writelines(lines[step_line_ind + 1 :])
	engine.run()
	if engine.has_completed():
		write_original(filename)
		return
	elif ct >= uset.recursion_depth:
		write_original(filename)
		return
	else:
		refine_run(ct)
