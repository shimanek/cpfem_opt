import numpy as np

def combine_error(errors):
	mean = np.mean(errors)
	if len(errors) == 1:
		return mean
	std = np.std(errors, ddof=1)
	combined = mean + 0.10 * std
	return combined
