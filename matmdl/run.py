"""
Runnable module to start an optimization run.
All input should be in an `input.toml` file in the directory where this is called.
"""

import os

import numpy as np

from matmdl import engines as engine
from matmdl import objectives as objective

from .core import optimizer as optimizer
from .core import parallel as parallel
from .core import runner as runner
from .core import writer as writer
from .core.experimental import ExpData
from .core.parser import uset
from .core.state import state
from .core.utilities import warn


def main():
	"""
	Instantiate data structures, start optimization loop.

	Checks for single run option, which runs then exits.
	Checks if current process is part of a parallel pool.
	Checks if previous output should be reloaded.
	"""
	runner.check_single()
	parallel.check_parallel()
	runner.remove_out_files()
	global exp_data, in_opt
	exp_data = ExpData(uset.orientations)
	in_opt = optimizer.InOpt(uset.orientations, uset.params)
	opt = optimizer.instantiate(in_opt, uset)
	if uset.do_load_previous:
		opt = optimizer.load_previous(opt)
	engine.prepare()

	loop(opt, uset.loop_len)


def loop(opt, loop_len):
	"""Holds all optimization iteration instructions."""

	def single_loop(opt):
		"""
		Run single iteration (one parameter set) of the optimization scheme.

		Single loops need to be separate function calls to allow empty returns to exit one
		parameter set.
		"""
		next_params = optimizer.get_next_param_set(opt, in_opt)
		writer.write_input_params(
			uset.param_file,
			in_opt.material_params,
			next_params[0 : in_opt.num_params_material],
		)

		with state.TimeRun()():
			for orient in in_opt.orients:
				engine.pre_run(next_params, orient, in_opt)

				engine.run()

				if not engine.has_completed():  # try decreasing max increment size
					runner.refine_run()
				if (
					not engine.has_completed()
				):  # if it still fails, tell optimizer a large error, continue
					opt.tell(next_params, uset.large_error)
					warn(
						f"Warning: early incomplete run for {orient}, skipping to next paramter set",
						RuntimeWarning,
					)
					return
				else:
					output_fname = f"temp_time_disp_force_{orient}.csv"
					if os.path.isfile(output_fname):
						os.remove(output_fname)
					engine.extract(orient)  # extract data to temp_time_disp_force.csv
					if np.sum(np.loadtxt(output_fname, delimiter=",", skiprows=1)[:, 1:2]) == 0:
						opt.tell(next_params, uset.large_error)
						warn(
							f"Warning: early incomplete run for {orient}, skipping to next paramter set",
							RuntimeWarning,
						)
						return

		# write out:
		update_params, update_errors = [], []
		with parallel.Checkout("out"):
			# check parallel instances:
			update_params_par, update_errors_par = parallel.update_parallel()
			if len(update_errors_par) > 0:
				update_params = update_params + update_params_par
				update_errors = update_errors + update_errors_par

			# this instance:
			errors = []
			for orient in in_opt.orients:
				errors.append(objective.calc_error(exp_data.data[orient]["raw"], orient))
				writer.combine_SS(zeros=False, orientation=orient)  # save stress-strain data

			combined_error = np.mean(errors)
			combined_error = objective.combine_error(errors)
			update_params = update_params + [next_params]
			update_errors = update_errors + [combined_error]

			# write this instance to file:
			writer.write_error_to_file(errors, in_opt.orients, objective.combine_error)
			writer.write_params_to_file(next_params, in_opt.params)

		# update optimizer outside of Checkout context to lower time using output files:
		optimizer.update_if_needed(opt, update_params, update_errors)

	runner.get_first(opt, in_opt, exp_data)
	for _ in range(loop_len):
		single_loop(opt)


if __name__ == "__main__":
	main()
