import unittest
from matmdl.parser import uset
import numpy as np


class TestError(unittest.TestCase):
	def test_stress_diff(self):
		from matmdl.objectives.rmse import _stress_diff, _slope_diff
		x = np.linspace(0.01, 1.0, 100)
		def curve1(x):
			return 5*x
		def curve2(x):
			return 10*x
		# def curve1(x):
		# 	return 3000*x + 7500*x**2
		# def curve2(x):
		# 	return 3000*x + 7500*x**2
		diff_stress = _stress_diff(x, curve1, curve2)
		diff_slope = _slope_diff(x, curve1, curve2)
		self.assertTrue(diff_slope - 50 < 1e-6)
		self.assertTrue(diff_stress - 50 < 1e-6)
		# print(" ")
		# print(diff_stress)
		# print(diff_slope)
		def curve1(x):
			return 5*x
		def curve2(x):
			return 5*x + 2
		diff_stress = _stress_diff(x, curve1, curve2)
		diff_slope = _slope_diff(x, curve1, curve2)
		self.assertTrue(diff_slope < 1e-6)
		# print(" ")
		# print(diff_stress)
		# print(diff_slope)		



class TestInput(unittest.TestCase):
	def test_input(self):
		self.assertTrue(uset.params['Tau0'] == [100,200])
		self.assertTrue(uset.orientations['001']['inp'] == 'mat_orient_100.inp')


class TestCP(unittest.TestCase):
	def test_rot(self):
		from matmdl.crystalPlasticity import get_offset_angle
		dir_in_load = np.array([1,2,3])
		dir_in_0deg = np.array([7,8,9])
		angle_in = 8.64489
		factor_out = get_offset_angle(dir_in_load, dir_in_0deg, angle_in)
		self.assertTrue(factor_out - 0.3 < 1e-6)


if __name__ == "__main__":
	unittest.main()
