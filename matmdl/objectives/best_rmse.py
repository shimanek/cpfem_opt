"""
Calculate the RMSE for the best parameter set.

Best parameters determined by objective function that wrote to out_errors.txt.
Use root mean squared error so that error value is more interpretable.
"""
import os

import numpy as np

from matmdl.core.experimental import ExpData
from matmdl.core.parser import uset
from matmdl.objectives import calc_error


def best_rmse():
	## load errors, find lowest
	errors = np.loadtxt(os.path.join(os.getcwd(), "out_errors.txt"), skiprows=1, delimiter=",")[:, -1]
	loc_min_error = np.argmin(errors)

	## from each orientation, collect best FD data
	orients = sorted(list(uset.orientations.keys()))
	data = {}
	for orient in orients:
		temp = np.load(f"out_time_disp_force_{orient}.npy", allow_pickle=False)
		data[orient] = temp[:,1:,loc_min_error]

	## set error slope weight to zero:
	with uset.unlock():
		uset.slope_weight = 0.0

	## call calc_error:
	exp_data = ExpData(uset.orientations)
	errors = {}
	for orient in orients:
		errors[orient] = calc_error(exp_data.data[orient]["raw"], orient, sim_data=data[orient])

	## write to stdout:
	print("\nRMSE values:", flush=True)
	for orient in  orients:
		name = orient+":"
		print(f"{name:7} {errors[orient]:10f}", flush=True)
	print(f"mean:   {np.mean(list(errors.values())):10f}\n", flush=True)

if __name__ == "__main__":
	best_rmse()
