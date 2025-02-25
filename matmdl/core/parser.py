"""
Module that loads and checks input file.
"""

import datetime
import os
from contextlib import contextmanager
from pprint import pprint

import tomllib

from matmdl.core.utilities import log


class UserSettings:
	"""
	Load, check, and store input from the input file.

	Note:
		Attributes must be written/deleted within an unlock context
		manager and should not be overwitten during the optimization
		since changing behavior makes the history harder to follow.
	"""

	class Option:
		"""Options that are commonly associated with each input."""

		def __init__(self, **kwargs):
			"""Defaults for option instances."""
			if "crit" in kwargs:
				self.crit = kwargs.get("crit")
			else:
				self.crit = True
			if "types" in kwargs:
				self.types = kwargs.get("types")
			else:
				self.types = []
			if "lower" in kwargs:
				self.lower = kwargs.get("lower")
			else:
				self.lower = None
			if "upper" in kwargs:
				self.upper = kwargs.get("upper")
			else:
				self.upper = None
			if "default" in kwargs:
				self.default = kwargs.get("default")

	input_reqs = {
		"run": {
			"loop_len": Option(types=[int], lower=2),
			"n_initial_points": Option(types=[int], lower=2),
			"large_error": Option(types=[int, float]),
			"param_file": Option(types=[str]),
			"length": Option(types=[int, float]),
			"area": Option(types=[int, float]),
			"jobname": Option(types=[str]),
			"recursion_depth": Option(types=[int]),
			"max_strain": Option(types=[int, float], crit=False, default=0.0),
			"min_strain": Option(types=[int, float], crit=False, default=0.0),
			"i_powerlaw": Option(types=[int]),
			"umat": Option(types=[str, bool], crit=False, default=False),
			"cpus": Option(types=[int]),
			"do_load_previous": Option(types=[bool, int]),
			"is_compression": Option(types=[bool]),
			"slope_weight": Option(types=[int, float], crit=False, default=0.4),
			"main_path": Option(types=[str], crit=False, default=os.getcwd()),
			"format": Option(types=[str], crit=False, default="huang"),
			"executable_path": Option(types=[str, bool], crit=False, default=False),
			"error_deviation_weight": Option(types=[float], crit=False, default=0.10, lower=0.0, upper=1.0),
			"do_single": Option(types=[bool], crit=False, default=False),
		},
		"plot": {
			"grain_size_name": Option(crit=False, types=[str]),
			"title": Option(crit=False, types=[str]),
			"param_additional_legend": Option(crit=False, types=[str]),
		},
	}

	def __init__(self, input_fname="input.toml"):
		categories = ["run", "plot"]
		with open(input_fname, "rb") as f:
			conf = tomllib.load(f)

		# write params:
		with self.unlock():
			self.params = conf["params"]
			if len(conf["orientations"]) > 0:
				self.orientations = {}
				for orient in conf["orientations"]:
					self.orientations[orient["name"]] = orient

			# get all input:
			for category in categories:
				for key, value in conf[category].items():
					if key not in self.input_reqs[category].keys():
						raise AttributeError(f"Unknown input: {key}")
					self.__dict__[key] = value

			# check if defaults needed:
			for category in categories:
				for key, value in self.input_reqs[category].items():
					if key not in self.__dict__:
						try:
							print(
								f"Input warning: input {key} not found, using default value of {value.default}"
							)
							self.__dict__[key] = value.default
						except AttributeError:
							raise AttributeError(f"\nInput: no default found for option {key}\n")

		# general checks:
		for key, req in self.input_reqs["run"].items():
			if key not in self.__dict__.keys():
				if req.crit is True:
					raise AttributeError(f"Missing critical input: {key}")
				else:
					continue
			value = self.__dict__[key]
			if req.types:
				input_type = type(value)
				if input_type not in req.types:
					raise AttributeError(f"Input type of {input_type} not one of {req.types}")
			if req.lower:
				if value < req.lower:
					raise ValueError(
						f"Input of {value} for `{key}` is below lower bound of {req.lower}"
					)
			if req.upper:
				value = self.__dict__[key]
				if value > req.upper:
					raise ValueError(
						f"Input of {value} for `{key}` is above upper bound of {req.upper}"
					)

		# check if this is a single run
		any_bounds = False
		for param_name, param_value in self.params.items():
			if type(param_value) in [list, tuple]:
				any_bounds = True
		with self.unlock():
			if not any_bounds:
				self.do_single = True
				log("Warning: parser: no bounded parameters in input file, running single.")
			else:
				self.do_single = False

		# individual checks:
		if self.i_powerlaw not in [0, 1]:
			raise NotImplementedError(f"No known option for i_powerlaw: {self.i_powerlaw}")
		if self.n_initial_points > self.loop_len:
			raise ValueError(
				f"Input initial points ({self.n_initial_points}) greater than total iterations ({self.loop_len})"
			)
		if self.format.lower() not in ["huang", "fepx"]:
			raise ValueError(
				f"Unexpected format option {self.format}; should be either huang or fepx."
			)
		# TODO add more individual checks if needed

	@contextmanager
	def unlock(self):
		self.is_locked = False
		try:
			yield self
		finally:
			self.is_locked = True

	def __setattr__(self, name, value):
		if name == "is_locked" or self.is_locked is False:
			super().__setattr__(name, value)
		else:
			raise AttributeError(self, "Don't touch my stuff")

	def __delattr__(self, name, value):
		if self.is_locked is False:
			super().__delattr__(name, value)
		else:
			raise AttributeError(self, "Don't wreck my stuff")


# instance for export:
date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
uset = UserSettings()

if __debug__:
	print(f"# matmdl printing input at {date_string}:")
	pprint(vars(uset))
	print("# end matmdl input")
