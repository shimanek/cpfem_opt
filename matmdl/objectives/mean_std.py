import numpy as np
from matmdl.core.parser import uset


def combine_error(errors):
	mean = np.mean(errors)
	std = np.std(errors, ddof=1)
	combined = mean + uset.error_deviation_weight * std
	return combined
