"""
Module for instantiating and updating the optimizer object.
"""

import os
import time

import numpy as np
from skopt import Optimizer

from matmdl.core.parser import uset
from matmdl.core.state import state
from matmdl.core.utilities import as_float_tuples, log, round_sig
from matmdl.core.writer import write_input_params


class InOpt:
	"""
	Stores information about the optimization input parameters.

	Since the hardening parameters and orientation parameters are fundamentally
	different, this object stores information about both in such a way that they
	can still be access independently.

	Args:
	    orientations (dict): Orientation information directly from ``opt_input``.
	    params (dict): name and bounds of parameters to be optimized.


	Attributes:
	    orients (list): Nickname strings defining orientations, as given
	        in :ref:`orientations`.
	    material_params (list): Parameter names to be optimized, as in :ref:`orientations`.
	    material_bounds (list): Tuple of floats defining bounds of parameter in the same
	        index of ``self.params``, again given in :ref:`orientations`.
	    orient_params (list): Holds orientation parameters to be optimized, or single
	        orientation parameters if not given as a tuple in :ref:`orientations`.
	        These are labeled ``orientationNickName_deg`` for the degree value of the
	        right hand rotation about the loading axis and ``orientationNickName_mag``
	        for the magnitude of the offset.
	    orient_bounds (list): List of tuples corresponding to the bounds for the parameters
	        stored in ``self.orient_params``.
	    params (list): Combined list consisting of both ``self.material_params`` and
	        ``self.orient_params``.
	    bounds (list): Combined list consisting of both ``self.material_bounds`` and
	        ``self.orient_bounds``.
	    has_orient_opt (dict): Dictionary with orientation nickname as key and boolean
	        as value indicating whether slight loading offsets should be considered
	        for that orientation.
	    fixed_vars (dict): Dictionary with orientation nickname as key and any fixed
	        orientation information (``_deg`` or ``_mag``) for that loading orientation
	        that is not going to be optimized.
	    offsets (list): List of dictionaries containing all information about the offset
	        as given in the input file. Not used/called anymore?
	    num_params_material (int): Number of material parameters to be optimized.
	    num_params_orient (int): Number of orientation parameters to be optimized.
	    num_params_total (int): Number of parameters to be optimized in total.

	Note:
	    Checks if ``orientations[orient]['offset']['deg_bounds']``
	    in :ref:`orientations` is a tuple to determine whether
	    orientation should also be optimized.
	"""

	# TODO: check if ``offsets`` attribute is still needed.
	def __init__(self, orientations, params):
		"""Sorted orientations here defines order for use in single list passed to optimizer."""
		self.orients = sorted(orientations.keys())
		(
			self.params,
			self.bounds,
			self.material_params,
			self.material_bounds,
			self.orient_params,
			self.orient_bounds,
		) = ([] for i in range(6))
		for param, bound in params.items():
			if type(bound) in (list, tuple):  # pass ranges to optimizer
				self.material_params.append(param)
				self.material_bounds.append([float(b) for b in bound])
			elif type(bound) in (float, int):  # write single values to file
				write_input_params(uset.param_file, param, float(bound))
			else:
				raise TypeError("Incorrect bound type in input file.")

		# add orientation offset info:
		self.offsets = []
		self.has_orient_opt = {}
		self.fixed_vars = {}
		for orient in self.orients:
			if "offset" in orientations[orient].keys():
				self.has_orient_opt[orient] = True
				self.offsets.append({orient: orientations[orient]["offset"]})
				# ^ saves all info (TODO: check if still needed)

				# deg rotation *about* loading orientation:
				if isinstance(orientations[orient]["offset"]["deg_bounds"], (tuple, list)):
					self.orient_params.append(orient + "_deg")
					self.orient_bounds.append(
						[float(f) for f in orientations[orient]["offset"]["deg_bounds"]]
					)
				else:
					self.fixed_vars[(orient + "_deg")] = orientations[orient]["offset"][
						"deg_bounds"
					]

				# mag rotation *away from* loading:
				if isinstance(orientations[orient]["offset"]["mag_bounds"], (tuple, list)):
					self.orient_params.append(orient + "_mag")
					self.orient_bounds.append(
						[float(f) for f in orientations[orient]["offset"]["mag_bounds"]]
					)
				else:
					self.fixed_vars[(orient + "_mag")] = orientations[orient]["offset"][
						"mag_bounds"
					]

			else:
				self.has_orient_opt[orient] = False

		# combine material and orient info into one ordered list:
		self.params = self.material_params + self.orient_params
		self.bounds = as_float_tuples(self.material_bounds + self.orient_bounds)

		# descriptive stats on input object:
		self.num_params_material = len(self.material_params)
		self.num_params_orient = len(self.orient_params)
		self.num_params_total = len(self.params)


def instantiate(in_opt: object, uset: object) -> object:
	"""
	Define all optimization settings, return optimizer object.

	Args:
	    in_opt: Input settings defined in :class:`InOpt`.
	    uset : User settings from input file.

	Returns:
	    skopt.Optimize: Instantiated optimization object.
	"""
	# Gaussian process with Matérn kernel as surrogate model
	from sklearn.gaussian_process.kernels import (
		RBF,
		ConstantKernel,
		DotProduct,
		ExpSineSquared,
		Matern,
		RationalQuadratic,
	)
	from skopt.learning import GaussianProcessRegressor
	from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
	noise_level = 0.1
	kernels = [
		1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
		1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
		1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
			length_scale_bounds=(0.1, 10.0),
			periodicity_bounds=(1.0, 10.0)),
		ConstantKernel(0.1, (0.01, 10.0))
			* (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
		1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
			nu=2.5),
	]
	kernel = kernels[0]
	gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise_level ** 2,
		normalize_y=True, noise="gaussian",
		n_restarts_optimizer=2
	)
	opt = Optimizer(
		dimensions=in_opt.bounds,
		base_estimator=gpr,
		n_initial_points=uset.n_initial_points,
		initial_point_generator="lhs",
		acq_func="EI",
		acq_func_kwargs={"xi": 1.0},  # default is 0.01, higher values favor exploration
	)
	return opt


def update_if_needed(opt, in_params, in_errors):
	"""
	Give params and errors to state, updating optimizer if needed.

	Need is determined by state.num_paramsets, which is set by relative
	timing of the FEA and opt.tell() procedures.

	`in_params` and `state.last_params` may contain duplicate updates from
	parallel instances, but these will be dealt with by opt.tell()
	"""
	if len(state.next_params) < 1:
		# tell optimizer all accumulated params and errors and clear from state
		update_params = []
		update_errors = []
		# check for old data stored in state:
		for params, errors in zip(state.last_params, state.last_errors):
			update_params.append(params)
			update_errors.append(errors)
		# add current information from args:
		update_params.append(in_params[0])
		update_errors.append(in_errors[0])
		# tell opt and clear state:
		with state.TimeTell()():
			opt.tell(update_params, update_errors)
		state.last_params = []
		state.last_errors = []
	else:
		# tell state of the params and error value
		state.last_params.append(in_params[0])
		state.last_errors.append(in_errors[0])


def get_next_param_set(opt: object, in_opt: object) -> list[float]:
	"""
	Give next parameter set to try using current optimizer state.

	Allow to sample bounds exactly, round all else to reasonable precision.
	"""
	if len(state.next_params) < 1:
		raw_param_list = opt.ask(n_points=state.num_paramsets)
		for raw_params in raw_param_list:
			new_params = []
			for param, bound in zip(raw_params, in_opt.bounds):
				if param in bound:
					new_params.append(param)
				else:
					new_params.append(round_sig(param, sig=6))
			state.next_params.append(new_params)
	new_params = state.next_params.popleft()
	return new_params


def load_previous(opt: object, search_local: bool = False) -> object:
	"""
	Load input files of previous optimizations to use as initial points in current optimization.

	Looks for a file named ``out_progress.txt`` from which to load previous results.
	Requires access to global variable ``opt_progress`` that stores optimization output.
	The parameter bounds for the input files must be within current parameter bounds.
	Renumbers old/loaded results in ``opt_progress`` to have negative iteration numbers.

	Args:
	    opt: Current instance of the optimizer object.
	    search_local: Look in the current directory for files
	        (convenient for plotting from parallel instances).

	Returns:
	    skopt.Optimizer: Updated instance of the optimizer object.
	"""
	fname_params = "out_progress.txt"
	fname_errors = "out_errors.txt"

	if uset.main_path not in [os.getcwd(), "."] and not search_local:
		fname_params = os.path.join(uset.main_path, fname_params)
		fname_errors = os.path.join(uset.main_path, fname_errors)

	params = np.loadtxt(fname_params, skiprows=1, delimiter=",")
	errors = np.loadtxt(fname_errors, skiprows=1, delimiter=",")
	x_in = params[:, 1:].tolist()
	y_in = errors[:, -1].tolist()

	if __debug__:
		with open("out_debug.txt", "a+") as f:
			f.write("loading previous results\n")
			f.writelines([f"x_in: {x}\ty_in: {y}\n" for x, y in zip(x_in, y_in)])

	tic = time.time()
	log("Starting to reload previous data")
	opt.tell(x_in, y_in)
	log(f"Finished reloading previous data after {time.time()-tic:.2f} seconds.")
	return opt
