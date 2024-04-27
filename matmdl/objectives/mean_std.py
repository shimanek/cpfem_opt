import numpy as np

from matmdl.core.parser import uset


def combine_error(errors):
	alpha = uset.error_deviation_weight
	mean = np.mean(errors)

	if len(errors) > 1:
		std = np.std(errors, ddof=1)
	else:
		std = 1.0

	combined = (1 - alpha) * mean + alpha * std
	return combined
